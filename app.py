import re
import json
import os
import ollama
import gradio as gr
import sympy as sp
from sympy.parsing.latex import parse_latex
from pydantic import BaseModel
from PIL import Image, ImageOps

# Explicit client connecting to loopback to prevent 0.0.0.0 connection failures on Windows
ollama_client = ollama.Client(host=os.environ.get("OLLAMA_API_HOST", "http://127.0.0.1:11434"))

# 1. Define your Structured Output Schema
class MathToolOutput(BaseModel):
    raw_ocr_latex: str
    cleaned_expression: str
    operation_type: str
    exact_result: str
    numeric_approximation: str
    status: str

# 2. The Deterministic Math Tool (SymPy Engine)
def execute_math_tool(latex_str: str) -> MathToolOutput:
    # Strip markdown code blocks and LaTeX math delimiters ($, $$, \[, \])
    cleaned = re.sub(r"(\$\$|\$|\\\[|\\\]|```latex|```)", "", latex_str).strip()
    # Remove trailing '=' or '= ?' if written on the canvas (e.g. "2 + 3 =")
    cleaned = re.sub(r"=\s*\??\s*$", "", cleaned).strip()

    try:
        # Handle Equations (e.g., "2x + 3 = 7") vs Expressions (e.g., "2 + 3" or "\int_0^2 x^2 dx")
        if "=" in cleaned:
            lhs_str, rhs_str = cleaned.split("=", 1)
            lhs = parse_latex(lhs_str)
            rhs = parse_latex(rhs_str)
            eq = sp.Eq(lhs, rhs)
            solutions = sp.solve(eq)
            
            # Format numerical approximations for numeric solutions
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
            # Parse LaTeX into a symbolic AST (Abstract Syntax Tree)
            expr = parse_latex(cleaned)
            
            # Evaluate calculus constructs if present (integrals, limits, derivatives, sums)
            if hasattr(expr, "doit"):
                evaluated = expr.doit()
            else:
                evaluated = expr
                
            exact_val = sp.simplify(evaluated)
            
            # Numeric evaluation
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

# 3. End-to-End Pipeline: Canvas Image -> glm-ocr -> Math Tool
def process_canvas(canvas_data):
    if canvas_data is None:
        return "Draw an equation first!", "{}"

    # Handle different Gradio canvas formats (dict with composite, or PIL Image)
    img = None
    if isinstance(canvas_data, dict):
        img = canvas_data.get("composite") or canvas_data.get("background")
    elif isinstance(canvas_data, Image.Image):
        img = canvas_data

    if img is None:
        return "Draw an equation first!", "{}"

    # Ensure RGB on pure white background (in case of RGBA with transparent canvas)
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    # Add white padding around edges (engineering golden rule for vision models)
    img = ImageOps.expand(img, border=25, fill=(255, 255, 255))

    img_path = "temp_equation.png"
    img.save(img_path)

    # Step A: Call glm-ocr via Ollama (Vision -> LaTeX)
    try:
        response = ollama_client.chat(
            model="glm-ocr",
            messages=[{
                "role": "user",
                "content": "Formula Recognition:",
                "images": [img_path]
            }],
            options={"temperature": 0.0}
        )
        raw_latex = response["message"]["content"].strip()
    except Exception as e:
        err_res = MathToolOutput(
            raw_ocr_latex="",
            cleaned_expression="",
            operation_type="ocr_error",
            exact_result="Ollama OCR failed",
            numeric_approximation="N/A",
            status=f"Ollama Error: {str(e)}"
        )
        return f"### OCR Error:\n{str(e)}", err_res.model_dump_json(indent=2)

    # Step B: Pass OCR output into our deterministic Math Tool
    structured_result = execute_math_tool(raw_latex)

    # Format nice LaTeX display + JSON structured tool output
    display_md = (
        f"### Detected LaTeX:\n$${structured_result.raw_ocr_latex}$$\n\n"
        f"### Result: **{structured_result.numeric_approximation}** "
        f"*(Exact: `{structured_result.exact_result}`)*"
    )
    return display_md, structured_result.model_dump_json(indent=2)

# 4. Launch Drawing UI (Works with your Laptop Pen/Touch or Mouse)
with gr.Blocks(title="Handwritten Math OCR + SymPy Calculator") as demo:
    gr.Markdown("## Handwritten Math OCR + Symbolic Tool Calculator (`glm-ocr` + `SymPy`)")
    gr.Markdown("Draw your handwritten formula or equation below using your pen, stylus, touch, or mouse.")
    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                type="pil",
                label="Draw Equation Here (Use Pen or Mouse)",
                canvas_size=(600, 300),
                brush=gr.Brush(default_size=4, colors=["#000000"], color_mode="fixed")
            )
            solve_btn = gr.Button("Read & Calculate (Run Pipeline)", variant="primary", size="lg")
        with gr.Column(scale=1):
            rendered_output = gr.Markdown("### Result will appear here...")
            json_output = gr.Code(language="json", label="Structured Tool Output (JSON)")

    solve_btn.click(fn=process_canvas, inputs=canvas, outputs=[rendered_output, json_output])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
