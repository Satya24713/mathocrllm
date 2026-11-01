import re
import json
import os
import numpy as np
import ollama
import gradio as gr
import sympy as sp
from sympy.parsing.latex import parse_latex
from pydantic import BaseModel
from PIL import Image, ImageOps
from texteller import load_model, load_tokenizer, img2latex

# Explicit client connecting to loopback to prevent 0.0.0.0 connection failures on Windows
ollama_client = ollama.Client(host=os.environ.get("OLLAMA_API_HOST", "http://127.0.0.1:11434"))

# Lazy loader for TexTeller 3.0
_texteller_model = None
_texteller_tok = None

def get_texteller():
    global _texteller_model, _texteller_tok
    if _texteller_model is None or _texteller_tok is None:
        print("Initializing TexTeller 3.0 model & tokenizer...")
        _texteller_tok = load_tokenizer()
        _texteller_model = load_model(use_onnx=True)
        print("TexTeller 3.0 ready.")
    return _texteller_model, _texteller_tok

def segment_formula_lines(pil_img, min_gap_rows=6, min_line_height=14):
    gray = np.array(pil_img.convert("L"))
    ink_mask = (gray < 200)
    row_ink = ink_mask.sum(axis=1)

    total_ink = row_ink.sum()
    if total_ink < 40:
        return [pil_img]

    noise_thresh = max(2, int(0.015 * row_ink.max()))
    is_ink_row = row_ink > noise_thresh

    bands = []
    in_band = False
    start_y = 0

    for y, has_ink in enumerate(is_ink_row):
        if has_ink and not in_band:
            in_band = True
            start_y = y
        elif not has_ink and in_band:
            in_band = False
            if (y - start_y) >= min_line_height:
                bands.append((start_y, y))
    if in_band and (len(is_ink_row) - start_y) >= min_line_height:
        bands.append((start_y, len(is_ink_row)))

    if len(bands) <= 1:
        return [pil_img]

    merged_bands = [bands[0]]
    for b in bands[1:]:
        prev_start, prev_end = merged_bands[-1]
        curr_start, curr_end = b
        gap = curr_start - prev_end
        if gap < min_gap_rows:
            merged_bands[-1] = (prev_start, curr_end)
        else:
            merged_bands.append(b)

    if len(merged_bands) <= 1:
        return [pil_img]

    w, h = pil_img.size
    crops = []
    for (y1, y2) in merged_bands:
        pad = 20
        crop_y1 = max(0, y1 - pad)
        crop_y2 = min(h, y2 + pad)
        cropped = pil_img.crop((0, crop_y1, w, crop_y2))
        cropped_padded = ImageOps.expand(cropped, border=15, fill="white")
        crops.append(cropped_padded)

    return crops

def recognize_latex_with_engine(img: Image.Image, engine_choice: str) -> tuple[str, str]:
    if "TexTeller" in engine_choice:
        model, tok = get_texteller()
        padded_img = ImageOps.expand(img, border=25, fill="white")
        line_crops = segment_formula_lines(padded_img)

        if len(line_crops) > 1:
            crop_arrays = [np.array(c.convert("RGB")) for c in line_crops]
            try:
                preds = img2latex(model, tok, crop_arrays)
                cleaned_lines = []
                for p in preds:
                    p_clean = re.sub(r"(\\\[|\\\]|\$\$|\$)", "", p).strip()
                    if p_clean:
                        cleaned_lines.append(p_clean)
                if len(cleaned_lines) >= 2:
                    return " \\\\ ".join(cleaned_lines), "TexTeller 3.0 (ONNX)"
                elif len(cleaned_lines) == 1:
                    return cleaned_lines[0], "TexTeller 3.0 (ONNX)"
            except Exception as e:
                print(f"Batched TexTeller multi-line fallback: {e}")

        full_arr = np.array(padded_img.convert("RGB"))
        preds = img2latex(model, tok, [full_arr])
        return (preds[0].strip() if preds else ""), "TexTeller 3.0 (ONNX)"
    else:
        # GLM-OCR via Ollama
        padded_img = ImageOps.expand(img, border=25, fill="white")
        img_path = "temp_equation.png"
        padded_img.save(img_path)
        response = ollama_client.chat(
            model="glm-ocr",
            messages=[{
                "role": "user",
                "content": "Formula Recognition:",
                "images": [img_path]
            }],
            options={"temperature": 0.0}
        )
        return response["message"]["content"].strip(), "GLM-OCR (Ollama)"

# 1. Define Structured Output Schema
class MathToolOutput(BaseModel):
    raw_ocr_latex: str
    cleaned_expression: str
    operation_type: str
    exact_result: str
    numeric_approximation: str
    status: str

# 2. Deterministic Math Tool (SymPy Engine)
def execute_math_tool(latex_str: str) -> MathToolOutput:
    cleaned = re.sub(r"(\$\$|\$|\\\[|\\\]|```latex|```)", "", latex_str).strip()
    cleaned = re.sub(r"=\s*\??\s*$", "", cleaned).strip()

    try:
        # Handle systems of equations (e.g., "9x + 8y = 0 \\ 4x + 9y = 0")
        split_lines = [s.strip() for s in re.split(r"\\\\|\\newline|\n|;", cleaned) if s.strip()]
        eq_lines = [s for s in split_lines if "=" in s]
        if len(eq_lines) >= 2:
            eqs = []
            for s in eq_lines:
                lhs_str, rhs_str = s.split("=", 1)
                lhs = parse_latex(lhs_str.strip())
                rhs = parse_latex(rhs_str.strip())
                eqs.append(sp.Eq(lhs, rhs))
            solutions = sp.solve(eqs)
            return MathToolOutput(
                raw_ocr_latex=latex_str,
                cleaned_expression="; ".join(str(e) for e in eqs),
                operation_type="system_of_equations",
                exact_result=str(solutions),
                numeric_approximation=str(solutions),
                status="success"
            )

        # Handle Single Equations vs Expressions
        if "=" in cleaned:
            lhs_str, rhs_str = cleaned.split("=", 1)
            lhs = parse_latex(lhs_str)
            rhs = parse_latex(rhs_str)
            eq = sp.Eq(lhs, rhs)
            solutions = sp.solve(eq)
            
            approx_solutions = []
            for s in solutions:
                try:
                    if hasattr(s, "evalf") and s.is_number:
                        approx_solutions.append(round(float(s.evalf()), 6))
                    else:
                        approx_solutions.append(str(s))
                except Exception:
                    approx_solutions.append(str(s))

            return MathToolOutput(
                raw_ocr_latex=latex_str,
                cleaned_expression=f"{lhs} = {rhs}",
                operation_type="equation_solver",
                exact_result=str(solutions),
                numeric_approximation=str(approx_solutions),
                status="success"
            )
        else:
            expr = parse_latex(cleaned)
            evaluated = expr.doit() if hasattr(expr, "doit") else expr
            exact_val = sp.simplify(evaluated)
            
            if exact_val.is_number:
                num_val = exact_val.evalf()
                if hasattr(num_val, "is_Integer") and num_val.is_Integer:
                    num_str = str(int(num_val))
                else:
                    try:
                        num_str = f"{float(num_val):.6g}"
                    except (TypeError, ValueError):
                        num_str = str(num_val)
            else:
                num_str = str(exact_val)
            
            return MathToolOutput(
                raw_ocr_latex=latex_str,
                cleaned_expression=str(expr),
                operation_type="symbolic_evaluation",
                exact_result=str(exact_val),
                numeric_approximation=num_str,
                status="success"
            )
    except Exception as e:
        return MathToolOutput(
            raw_ocr_latex=latex_str,
            cleaned_expression=cleaned,
            operation_type="error",
            exact_result="Could not evaluate",
            numeric_approximation="N/A",
            status=f"Parse Error: {str(e)}"
        )

# 3. End-to-End Pipeline
def process_canvas(canvas_data, engine_choice="GLM-OCR (Ollama)"):
    if canvas_data is None:
        return "Draw an equation first!", "{}"

    img = None
    if isinstance(canvas_data, dict):
        img = canvas_data.get("composite") or canvas_data.get("background")
    elif isinstance(canvas_data, Image.Image):
        img = canvas_data

    if img is None:
        return "Draw an equation first!", "{}"

    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    try:
        raw_latex, engine_label = recognize_latex_with_engine(img, engine_choice)
    except Exception as e:
        err_res = MathToolOutput(
            raw_ocr_latex="",
            cleaned_expression="",
            operation_type="ocr_error",
            exact_result=f"{engine_choice} OCR failed",
            numeric_approximation="N/A",
            status=f"OCR Error: {str(e)}"
        )
        return f"### OCR Error ({engine_choice}):\n{str(e)}", err_res.model_dump_json(indent=2)

    structured_result = execute_math_tool(raw_latex)

    display_md = (
        f"### Detected LaTeX (`{engine_label}`):\n$${structured_result.raw_ocr_latex}$$\n\n"
        f"### Result: **{structured_result.numeric_approximation}** "
        f"*(Exact: `{structured_result.exact_result}`)*"
    )
    return display_md, structured_result.model_dump_json(indent=2)

# 4. Launch Drawing UI
with gr.Blocks(title="Handwritten Math OCR + SymPy Calculator") as demo:
    gr.Markdown("## Handwritten Math OCR + Symbolic Calculator (Dual Engine: `GLM-OCR` & `TexTeller 3.0`)")
    gr.Markdown("Select your preferred OCR engine, draw a formula on the whiteboard, and calculate exact symbolic results.")
    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                type="pil",
                label="Draw Equation Here (Use Pen or Mouse)",
                canvas_size=(600, 300),
                brush=gr.Brush(default_size=4, colors=["#000000"], color_mode="fixed")
            )
            engine_choice = gr.Radio(
                choices=["GLM-OCR (Ollama)", "TexTeller 3.0 (ONNX)"],
                value="GLM-OCR (Ollama)",
                label="🧠 OCR Engine",
                info="GLM-OCR: 0.9B Multimodal VLM | TexTeller 3.0: 298M ViT + Auto-Line Splitter"
            )
            solve_btn = gr.Button("Read & Calculate (Run Pipeline)", variant="primary", size="lg")
        with gr.Column(scale=1):
            rendered_output = gr.Markdown("### Result will appear here...")
            json_output = gr.Code(language="json", label="Structured Tool Output (JSON)")

    solve_btn.click(fn=process_canvas, inputs=[canvas, engine_choice], outputs=[rendered_output, json_output])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
