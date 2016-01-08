# This file is a part of Julia. License is MIT: http://julialang.org/license

# LQ Factorizations

immutable LQ{T,S<:AbstractMatrix} <: Factorization{T}
    factors::S
    τ::Vector{T}
    LQ(factors::AbstractMatrix{T}, τ::Vector{T}) = new(factors, τ)
end

immutable LQPackedQ{T} <: AbstractMatrix{T}
    factors::Matrix{T}
    τ::Vector{T}
end

LQ{T}(factors::AbstractMatrix{T}, τ::Vector{T}) = LQ{T,typeof(factors)}(factors, τ)

lqfact!{T<:BlasFloat}(A::StridedMatrix{T}) = LQ(LAPACK.gelqf!(A)...)
lqfact{T<:BlasFloat}(A::StridedMatrix{T})  = lqfact!(copy(A))
lqfact(x::Number) = lqfact(fill(x,1,1))

function lq(A::Union{Number, AbstractMatrix}; thin::Bool=true)
    F = lqfact(A)
    F[:L], full(F[:Q], thin=thin)
end

convert{T}(::Type{LQ{T}},A::LQ) = LQ(convert(AbstractMatrix{T}, A.factors), convert(Vector{T}, A.τ))
convert{T}(::Type{Factorization{T}}, A::LQ) = convert(LQ{T}, A)

function getindex(A::LQ, d::Symbol)
    m, n = size(A)
    if d == :L
        return tril!(A.factors[1:m, 1:min(m,n)])
    elseif d == :Q
        return LQPackedQ(A.factors,A.τ)
    else
        throw(KeyError(d))
    end
end

getindex(A::LQPackedQ, i::Integer, j::Integer) = (x = zeros(eltype(A), size(A, 1)); x[i] = 1; y = zeros(eltype(A), size(A, 2)); y[j] = 1; dot(x, A*y))

getq(A::LQ) = LQPackedQ(A.factors, A.tau)

convert{T}(::Type{LQPackedQ{T}}, Q::LQPackedQ) = LQPackedQ(convert(AbstractMatrix{T}, Q.factors), convert(Vector{T}, Q.τ))
convert{T}(::Type{AbstractMatrix{T}}, Q::LQPackedQ) = convert(LQPackedQ{T}, Q)

size(A::LQ, dim::Integer) = size(A.factors, dim)
size(A::LQ) = size(A.factors)
size(A::LQPackedQ, dim::Integer) = 0 < dim ? (dim <= 2 ? size(A.factors, 1) : 1) : throw(BoundsError())
size(A::LQPackedQ) = size(A, 1), size(A, 2)

full(A::LQ) = A[:L]*A[:Q]
function full{T}(A::LQPackedQ{T}; thin::Bool=true)
    if thin
        LAPACK.orglq!(copy(A.factors),A.τ)
    else
        A_mul_B!(A, eye(T, size(A.factors,2)))
    end
end

## Multiplication by Q
### QB
A_mul_B!{T<:BlasFloat}(A::LQPackedQ{T}, B::StridedVecOrMat{T})   = LAPACK.ormlq!('L','N',A.factors,A.τ,B)
function *{TA,TB}(A::LQPackedQ{TA},B::StridedVecOrMat{TB})
    TAB = promote_type(TA, TB)
    A_mul_B!(convert(AbstractMatrix{TAB}, A), TB==TAB ? copy(B) : convert(AbstractArray{TAB}, B))
end

### QcB
Ac_mul_B!{T<:BlasReal}(A::LQPackedQ{T}, B::StridedVecOrMat{T})    = LAPACK.ormlq!('L','T',A.factors,A.τ,B)
Ac_mul_B!{T<:BlasComplex}(A::LQPackedQ{T}, B::StridedVecOrMat{T}) = LAPACK.ormlq!('L','C',A.factors,A.τ,B)
function Ac_mul_B{TA,TB}(A::LQPackedQ{TA}, B::StridedVecOrMat{TB})
    TAB = promote_type(TA,TB)
    if size(B,1) == size(A.factors,2)
        if TB == TAB
            Ac_mul_B!(convert(AbstractMatrix{TAB}, A), copy(B))
        else
            Ac_mul_B!(convert(AbstractMatrix{TAB}, A), convert(AbstractMatrix{TAB}, B))
        end
    elseif size(B,1) == size(A.factors,1)
        Ac_mul_B!(convert(AbstractMatrix{TAB}, A), [B; zeros(TAB, size(A.factors, 2) - size(A.factors, 1), size(B, 2))])
    else
        throw(DimensionMismatch("first dimension of B, $(size(B,1)), must equal one of the dimensions of A, $(size(A))"))
    end
end

### AQ
A_mul_B!{T<:BlasFloat}(A::StridedMatrix{T}, B::LQPackedQ{T}) = LAPACK.ormlq!('R', 'N', B.factors, B.τ, A)
function *{TA,TB}(A::StridedMatrix{TA},B::LQPackedQ{TB})
    TAB = promote_type(TA,TB)
    if size(B.factors,2) == size(A,2)
        if TA == TAB
            A_mul_B!(copy(A),convert(AbstractMatrix{TAB},B))
        else
            A_mul_B!(convert(AbstractMatrix{TAB}, A),convert(AbstractMatrix{TAB},B))
        end
    elseif size(B.factors,1) == size(A,2)
        A_mul_B!( [A zeros(TAB, size(A,1), size(B.factors,2)-size(B.factors,1))], convert(AbstractMatrix{TAB},B))
    else
        throw(DimensionMismatch("second dimension of A, $(size(A,2)), must equal one of the dimensions of B, $(size(B))"))
    end
end

### AQc
A_mul_Bc!{T<:BlasReal}(A::StridedMatrix{T}, B::LQPackedQ{T})    = LAPACK.ormlq!('R','T',B.factors,B.τ,A)
A_mul_Bc!{T<:BlasComplex}(A::StridedMatrix{T}, B::LQPackedQ{T}) = LAPACK.ormlq!('R','C',B.factors,B.τ,A)
function A_mul_Bc{TA<:Number,TB<:Number}( A::StridedVecOrMat{TA}, B::LQPackedQ{TB})
    TAB = promote_type(TA,TB)
    if TA == TAB
        A_mul_Bc!(copy(A), convert(AbstractMatrix{TAB},B))
    else
        A_mul_Bc!(convert(AbstractArray{TAB}, A), convert(AbstractMatrix{TAB},B))
    end
end

function \{TA,Tb}(A::LQ{TA}, b::StridedVector{Tb})
    S = promote_type(TA,Tb)
    m,n = size(A)
    m == length(b) || throw(DimensionMismatch("left hand side has $m rows, but right hand side has length $(length(b))"))
    AA = convert(Factorization{S}, A)
    if n > m
        x = A_ldiv_B!(AA, [b; zeros(S, n - m)])
    else
        x = A_ldiv_B!(AA, copy_oftype(b, S))
    end
    return length(x) > n ? x[1:n] : x
end
function \{TA,TB}(A::LQ{TA},B::StridedMatrix{TB})
    S = promote_type(TA,TB)
    m,n = size(A)
    m == size(B,1) || throw(DimensionMismatch("left hand side has $m rows, but right hand side has $(size(B,1)) rows"))
    AA = convert(Factorization{S}, A)
    if n > m
        X = A_ldiv_B!(AA, [B; zeros(S, n - m, size(B, 2))])
    else
        X = A_ldiv_B!(AA, copy_oftype(B, S))
    end
    return size(X, 1) > n ? X[1:n,:] : X
end

function A_ldiv_B!{T}(A::LQ{T}, B::StridedVecOrMat{T})
    Ac_mul_B!(A[:Q], A_ldiv_B!(LowerTriangular(A[:L]),B))
    return B
end
