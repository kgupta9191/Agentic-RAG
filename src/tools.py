import re
import sympy as sp


def extract_equations(text: str):
    """
    Extract simple equations from retrieved text.
    This works best for plain-text equations like:
    F = m*a
    y = x**2 + 3*x
    m_a = rho*pi*R**2
    """

    patterns = [
        r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^.\n]+",
        r"\$([^$=]+=[^$]+)\$",
    ]

    equations = []

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            eq = match.strip()
            if len(eq) < 80:
                equations.append(eq)

    return list(set(equations))


def solve_equation_tool(equation: str, variable: str | None = None):
    """
    Solves symbolic equations using SymPy.
    Example:
        equation = "F = m*a"
        variable = "a"
    """

    try:
        equation = equation.replace("^", "**")

        if "=" not in equation:
            expr = sp.sympify(equation)
            simplified = sp.simplify(expr)
            return f"Simplified expression: {simplified}"

        lhs, rhs = equation.split("=", 1)

        lhs_expr = sp.sympify(lhs.strip())
        rhs_expr = sp.sympify(rhs.strip())

        eq = sp.Eq(lhs_expr, rhs_expr)

        if variable:
            var = sp.Symbol(variable)
            solution = sp.solve(eq, var)
            return f"Solution for {variable}: {solution}"

        symbols = list(eq.free_symbols)

        if not symbols:
            return f"Equation evaluated: {sp.simplify(lhs_expr - rhs_expr) == 0}"

        solutions = {}
        for sym in symbols:
            try:
                solutions[str(sym)] = sp.solve(eq, sym)
            except Exception:
                pass

        return f"Possible symbolic solutions: {solutions}"

    except Exception as e:
        return f"Could not solve equation. Error: {e}"


def equation_tool_from_context(context: str, user_question: str):
    equations = extract_equations(context)

    if not equations:
        return "No clear equation found in the retrieved context."

    variable = None

    variable_match = re.search(r"solve for\s+([A-Za-z_][A-Za-z0-9_]*)", user_question.lower())
    if variable_match:
        variable = variable_match.group(1)

    results = []

    for eq in equations[:5]:
        result = solve_equation_tool(eq, variable)
        results.append(f"Equation: {eq}\nResult: {result}")

    return "\n\n".join(results)
