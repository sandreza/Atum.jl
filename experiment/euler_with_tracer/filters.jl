function vandermonde(x, N)
    # create view to assign values
    P = zeros(length(x), N + 1)
    P⁰ = view(P, :, 0 + 1)
    @. P⁰ = 1

    # explicitly compute second coefficient
    if N == 0
        return P
    end

    P¹ = view(P, :, 1 + 1)
    @. P¹ = x

    if N == 1
        return P
    end

    for n in 1:(N-1)
        # get views for ith, i-1th, and i-2th columns
        Pⁿ⁺¹ = view(P, :, n + 1 + 1)
        Pⁿ = view(P, :, n + 0 + 1)
        Pⁿ⁻¹ = view(P, :, n - 1 + 1)

        # compute coefficients for ith column
        @. Pⁿ⁺¹ = ((2n + 1) * x * Pⁿ - n * Pⁿ⁻¹) / (n + 1)
    end

    return P
end

#=
# Legendre Polynomials
P₀(x) = 1
P₁(x) = x
P₂(x) = 0.5 * (3 * x^2 - 1)
P₃(x) = 0.5 * (5 * x^3 - 3 * x)
P₄(x) = 0.125 * (35 * x^4 - 30 * x^2 + 3)
P₅(x) = 0.125 * (63 * x^5 - 70 * x^3 + 15 * x)
=#

#=
x = Array(cpucell.points_1d[1])[:]
V⁻¹ = vandermonde(x, length(cpucell.points_1d[1]) - 1)
V = inv(V⁻¹)
M = V * 0 + I

M[end, end] = 1.0
filter = V⁻¹ * M * V
P = Bennu.Kron((A(filter), A(filter)))
function closure_overintegrate(filter)
    function overintegrate(_, _, q)
        q .= filter * q
    end
    return overintegrate
end
overintegrate = closure_overintegrate(P)
=#

