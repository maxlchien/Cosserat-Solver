module elastic_core
  use cosserat_kinds, only: rk
  implicit none
  private
  public :: greens_displacement_force

  real(rk), parameter :: pi = 3.141592653589793238462643383279502884197_rk
contains

  function identity(n) result(I)
    integer, intent(in) :: n
    integer :: k
    real(rk), allocatable :: I(:,:)
    allocate(I(n,n))
    I = 0.0_rk
    do k = 1, n
      I(k,k) = 1.0_rk
    end do
  end function identity

  function outer_product(a, b) result(C)
    real(rk), intent(in) :: a(:), b(:)
    real(rk), allocatable :: C(:,:)
    integer :: n, m, i, j
    n = size(a)
    m = size(b)
    allocate(C(n,m))
    C = reshape([((a(i)*b(j), i=1,n), j=1,m)], [n,m])
  end function outer_product

  function kelvin_static_green(r, r_hat, lam, mu) result(G)
    real(rk), intent(in) :: r, r_hat(3), lam, mu
    real(rk) :: G(3, 3)

    real(rk) :: factor, i_coeff, rr_coeff

    factor = 1.0_rk / (8.0_rk * pi * mu * (lam + 2.0_rk * mu) * r)
    i_coeff = lam + 3.0_rk * mu
    rr_coeff = lam + mu
    G = factor * (i_coeff * identity(3) + rr_coeff * outer_product(r_hat, r_hat))
  end function kelvin_static_green

  function hessian_eikr_over_r(k, r, r_hat) result(H)
    real(rk), intent(in) :: k ! this is defined for complex k but we don't need it
    real(rk), intent(in) :: r, r_hat(3)
    complex(rk) :: H(3, 3)

    complex(rk) :: expikr, psi, psi_prime_over_r, radial_coeff

    expikr = exp((0.0_rk, 1.0_rk) * k * r)
    psi = expikr / r
    psi_prime_over_r = ((0.0_rk, 1.0_rk) * k * r - 1.0_rk) * expikr / (r**3)
    radial_coeff = -(k**2) * psi - 3.0_rk * psi_prime_over_r
    H = psi_prime_over_r * identity(3) + radial_coeff * outer_product(r_hat, r_hat)

  end function hessian_eikr_over_r

  function greens_displacement_force(x, omega, rho, lam, mu) result(G)
    ! Form used:
    ! G = (1/(4*pi*mu)) * [ phi_s I + (1/k_s^2) (H_s - H_p) ]
    ! where phi_a = exp(i k_a r)/r and H_a = grad grad phi_a.
    real(rk), intent(in) :: x(3)
    real(rk), intent(in) :: omega, rho, lam, mu
    complex(rk) :: G(3, 3)

    real(rk) :: R
    real(rk) :: R_hat(3)

    real(rk) :: cp, cs, kp, ks
    complex(rk) :: phi_s, h_s(3, 3), h_p(3, 3)

    ! validation of parameters occurs at Python wrapper boundary

    R = sqrt(sum(x**2))
    R_hat = x / R

    if (abs(omega) < 1.0e-12_rk) then
      ! return early
      G = kelvin_static_green(R, R_hat, lam, mu)
      return
    end if



    cp = sqrt((lam + 2.0_rk * mu) / rho)
    cs = sqrt(mu / rho)

    kp = omega / cp
    ks = omega / cs

    phi_s = exp((0.0_rk, 1.0_rk) * ks * R) / R
    h_s = hessian_eikr_over_r(ks, R, R_hat)
    h_p = hessian_eikr_over_r(kp, R, R_hat)

    G = (1.0_rk / (4.0_rk * pi * mu)) * (phi_s * identity(3) + (h_s - h_p) / (ks**2))
  end function greens_displacement_force

end module elastic_core
