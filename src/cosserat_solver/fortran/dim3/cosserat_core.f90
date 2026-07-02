module cosserat_core
  use cosserat_kinds, only: rk
  use dispersion_core
  implicit none
  private
  public :: greens_mixed_force, greens_displacement_force, greens_rotation_force
  public :: greens_displacement_force_static, greens_rotation_force_static

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

  function greens_mixed_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(G)
    real(rk), intent(in) :: x(3)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: G(6, 6)

    G(1:6, 1:3) = greens_displacement_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    G(1:6, 4:6) = greens_rotation_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
  end function greens_mixed_force

  function greens_displacement_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(G)
    real(rk), intent(in) :: x(3)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: G(6, 3)

    real(rk) :: R
    real(rk) :: R_hat(3)

    real(rk) :: c2_sq, c4_sq, w0_sq
    complex(rk) :: k1_sq, k2_sq, k4_sq
    complex(rk) :: k1_val, k2_val, k4_val
    complex(rk) :: A1, A2, A4

    complex(rk) :: term1_prefactor
    complex(rk) :: term2_prefactor, term2_n1, term2_n2, term2_n4
    complex(rk) :: term3_prefactor, term3_n1, term3_n2, term3_n4
    complex(rk) :: rotation_term_prefactor

    complex(rk) :: term1(3, 3), term2(3, 3), term3(3, 3), rotation_term(3, 3)

    ! validation of parameters occurs at Python wrapper boundary

    if (omega == 0.0_rk) then
      ! return early
      G = greens_displacement_force_static(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
      return
    end if

    R = sqrt(sum(x**2))
    R_hat = x / R

    c2_sq = c2_squared(rho, mu, nu)
    c4_sq = c4_squared(J, mu_c, nu_c)
    w0_sq = w0_squared(nu, J)

    k1_sq = k1_squared(omega, rho, lam, mu)
    k2_sq = k2_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k4_sq = k4_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k1_val = k1(omega, rho, lam, mu)
    k2_val = k2(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k4_val = k4(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    A1 = 1.0_rk / ((k2_sq - k1_sq) * (k4_sq - k1_sq))
    A2 = 1.0_rk / ((k4_sq - k2_sq) * (k1_sq - k2_sq))
    A4 = 1.0_rk / ((k1_sq - k4_sq) * (k2_sq - k4_sq))

    ! -I_3 * \frac{\hat{f}_\omega}{r}\frac{\rho }{4\pi(\mu+\nu)}\frac{1}{k_4^2-k_2^2} *
    ! \paren{e^{ik_2r}\paren{\frac{\omega^2-\omega_0^2}{c_4^2}-k_2^2}-e^{ik_4r}\paren{\frac{\omega^2-\omega_0^2}{c_4^2}-k_4^2}}
    term1_prefactor = 1.0_rk / (4.0_rk * pi * (mu + nu))
    term1 = -term1_prefactor / (k4_sq - k2_sq) * &
        (exp((0.0_rk, 1.0_rk) * k2_val * R) * ((omega**2 - w0_sq) / c4_sq - k2_sq) - &
         exp((0.0_rk, 1.0_rk) * k4_val * R) * ((omega**2 - w0_sq) / c4_sq - k4_sq)) * &
        identity(3)

    ! I_3 * \frac{\rho\paren{\lambda+\mu-\nu}}{4\pi(\lambda+2\mu)(\mu+\nu)} *
    ! \sum_{n=1,2,4} A_n \paren{\frac{\omega^2-\omega_0^2}{c_4^2}-\frac{j\omega_0^2}{4c_2^2c_4^2}-k_n^2}\paren{ik_nr-1}\frac{e^{ik_nr}}{r^3}
    term2_prefactor = -(lam + mu - nu) / (4.0_rk * pi * (lam + 2.0_rk * mu) * (mu + nu))
    term2_n1 = ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4.0_rk * c2_sq * c4_sq) - k1_sq) * &
        ((0.0_rk, 1.0_rk) * k1_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k1_val * R) / R**3
    term2_n2 = ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4.0_rk * c2_sq * c4_sq) - k2_sq) * &
        ((0.0_rk, 1.0_rk) * k2_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k2_val * R) / R**3
    term2_n4 = ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4.0_rk * c2_sq * c4_sq) - k4_sq) * &
        ((0.0_rk, 1.0_rk) * k4_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k4_val * R) / R**3
    term2 = term2_prefactor * (A1 * term2_n1 + A2 * term2_n2 + A4 * term2_n4) * identity(3)

    ! - \hat{r} \hat{r}^T \frac{\rho\paren{\lambda+\mu-\nu}}{4\pi(\lambda+2\mu)(\mu+\nu)} *
    ! \sum_{n=1,2,4} A_n \paren{\frac{\omega^2-\omega_0^2}{c_4^2}-\frac{j\omega_0^2}{4c_2^2c_4^2}-k_n^2}\paren{-k_n^2r^2-3ik_nr+3} \frac{e^{ik_nr}}{r^3}
    term3_prefactor = -(lam + mu - nu) / (4.0_rk * pi * (lam + 2.0_rk * mu) * (mu + nu))
    term3_n1 = ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4.0_rk * c2_sq * c4_sq) - k1_sq) * &
        (-k1_sq * R**2 - 3.0_rk * (0.0_rk, 1.0_rk) * k1_val * R + 3.0_rk) * exp((0.0_rk, 1.0_rk) * k1_val * R) / R**3
    term3_n2 = ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4.0_rk * c2_sq * c4_sq) - k2_sq) * &
        (-k2_sq * R**2 - 3.0_rk * (0.0_rk, 1.0_rk) * k2_val * R + 3.0_rk) * exp((0.0_rk, 1.0_rk) * k2_val * R) / R**3
    term3_n4 = ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4.0_rk * c2_sq * c4_sq) - k4_sq) * &
        (-k4_sq * R**2 - 3.0_rk * (0.0_rk, 1.0_rk) * k4_val * R + 3.0_rk) * exp((0.0_rk, 1.0_rk) * k4_val * R) / R**3
    term3 = term3_prefactor * (A1 * term3_n1 + A2 * term3_n2 + A4 * term3_n4) * outer_product(R_hat, R_hat)

    ! [\hat{r}\times -] \frac{1}{r^2} \frac{\rho \nu}{2\pi(\mu+\nu)(\mu_c+\nu_c)}\frac{1}{k_4^2-k_2^2}\paren{(ik_2r-1)e^{ik_2r}-(ik_4r-1)e^{ik_4r}}
    rotation_term_prefactor = -nu / (2.0_rk * pi * (mu + nu) * (mu_c + nu_c)) / R**2
    rotation_term = rotation_term_prefactor / (k4_sq - k2_sq) * &
        ((0.0_rk, 1.0_rk) * k2_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k2_val * R) &
        - ((0.0_rk, 1.0_rk) * k4_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k4_val * R)
    rotation_term = rotation_term * reshape( &
      [R_hat(2), -R_hat(3), R_hat(1), &
      -R_hat(2), R_hat(3), -R_hat(1), &
      R_hat(3), -R_hat(1), R_hat(2)], [3,3])

    G(1:3, 1:3) = term1 + term2 + term3
    G(4:6, 1:3) = rotation_term
  end function greens_displacement_force

  function greens_displacement_force_static(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(G)
    real(rk), intent(in) :: x(3)
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: G(6, 3)

    real(rk) :: R
    real(rk) :: R_hat(3)

    R = sqrt(sum(x**2))
    R_hat = x / R

    ! TODO: derive the correct static Green's function. For now, return zeros
    G = 0.0_rk
  end function greens_displacement_force_static

  function greens_rotation_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(G)
    real(rk), intent(in) :: x(3)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: G(6, 3)

    real(rk) :: R
    real(rk) :: R_hat(3)

    real(rk) :: c2_sq, c3_sq, w0_sq
    complex(rk) :: k2_sq, k3_sq, k4_sq
    complex(rk) :: k2_val, k3_val, k4_val
    complex(rk) :: B2, B3, B4

    complex(rk) :: displacement_term_prefactor
    complex(rk) :: displacement_term_n2, displacement_term_n3, displacement_term_n4
    complex(rk) :: term1_prefactor, term1_n2, term1_n3, term1_n4
    complex(rk) :: term2_prefactor, term2_n2, term2_n3, term2_n4
    complex(rk) :: term3_prefactor, term3_n2, term3_n3, term3_n4
    complex(rk) :: term4_prefactor, term4_n2, term4_n3, term4_n4
    complex(rk) :: term5_prefactor, term5_n2, term5_n3, term5_n4

    complex(rk) :: displacement_term(3, 3), term1(3, 3), term2(3, 3), term3(3, 3), term4(3, 3), term5(3, 3)

    ! validation of parameters occurs at Python wrapper boundary

    if (omega == 0.0_rk) then
      ! return early
      G = greens_rotation_force_static(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
      return
    end if

    R = sqrt(sum(x**2))
    R_hat = x / R

    c2_sq = c2_squared(rho, mu, nu)
    c3_sq = c3_squared(J, lam_c, mu_c)
    w0_sq = w0_squared(nu, J)

    k2_sq = k2_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k3_sq = k3_squared(omega, nu, J, lam_c, mu_c)
    k4_sq = k4_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k2_val = k2(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k3_val = k3(omega, nu, J, lam_c, mu_c)
    k4_val = k4(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    B2 = 1.0_rk / (k3_sq - k2_sq) / (k4_sq - k2_sq)
    B3 = 1.0_rk / (k2_sq - k3_sq) / (k4_sq - k3_sq)
    B4 = 1.0_rk / (k2_sq - k4_sq) / (k3_sq - k4_sq)

    ! [\hat{r} \times -]\frac{1}{r^2}\frac{\rho j \nu}{2\pi(\mu+\nu)(\mu_c+\nu_c)} *
    ! \sum_{n=2,3,4}B_n \paren{\frac{\omega^2-\omega_0^2}{c_3^2}-k_n^2}\paren{ik_nr-1}e^{ik_nr}
    displacement_term_prefactor = -J * nu / (2.0_rk * pi * (mu + nu) * (mu_c + nu_c)) / R**2
    displacement_term_n2 = ((omega**2 - w0_sq) / c3_sq - k2_sq) * ((0.0_rk, 1.0_rk) * k2_val * R - 1.0_rk) &
      * exp((0.0_rk, 1.0_rk) * k2_val * R)
    displacement_term_n3 = ((omega**2 - w0_sq) / c3_sq - k3_sq) * ((0.0_rk, 1.0_rk) * k3_val * R - 1.0_rk) &
      * exp((0.0_rk, 1.0_rk) * k3_val * R)
    displacement_term_n4 = ((omega**2 - w0_sq) / c3_sq - k4_sq) * ((0.0_rk, 1.0_rk) * k4_val * R - 1.0_rk) &
      * exp((0.0_rk, 1.0_rk) * k4_val * R)
    displacement_term = displacement_term_prefactor * &
      (B2 * displacement_term_n2 + B3 * displacement_term_n3 + B4 * displacement_term_n4) * reshape( &
      [0.0_rk, -R_hat(3), R_hat(2), &
       R_hat(3), 0.0_rk, -R_hat(1), &
       -R_hat(2), R_hat(1), 0.0_rk], [3,3])

    ! -I_3\frac{1}{r} \frac{\rho j}{4\pi(\mu_c+\nu_c)}\sum_{n=2,3,4}B_n \paren{\frac{\omega^2}{c_2^2}-k_n^2} *
    ! \paren{\frac{\omega^2-\omega_0^2}{c_3^2}-k_n^2}e^{ik_nr}
    term1_prefactor = J / (4.0_rk * pi * (mu_c + nu_c)) / R
    term1_n2 = ((omega**2 / c2_sq - k2_sq) * ((omega**2 - w0_sq) / c3_sq - k2_sq) * exp((0.0_rk, 1.0_rk) * k2_val * R))
    term1_n3 = ((omega**2 / c2_sq - k3_sq) * ((omega**2 - w0_sq) / c3_sq - k3_sq) * exp((0.0_rk, 1.0_rk) * k3_val * R))
    term1_n4 = ((omega**2 / c2_sq - k4_sq) * ((omega**2 - w0_sq) / c3_sq - k4_sq) * exp((0.0_rk, 1.0_rk) * k4_val * R))
    term1 = term1_prefactor * (term1_n2 + term1_n3 + term1_n4) * identity(3)

    ! I_3\frac{\rho j(\lambda_c+\mu_c-\nu_c)}{4\pi(\lambda_c+2\mu_c)(\mu_c+\nu_c)} *
    ! \sum_{n=2,3,4}B_n\paren{\frac{\omega^2}{c_2^2}-k_n^2}\paren{ik_nr-1}\frac{e^{ik_nr}}{r^3}
    term2_prefactor = - J * (lam_c + mu_c - nu_c) / (4.0_rk * pi * (lam_c + 2.0_rk * mu_c) * (mu_c + nu_c))
    term2_n2 = ((omega**2 / c2_sq - k2_sq) * ((0.0_rk, 1.0_rk) * k2_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k2_val * R) / R**3)
    term2_n3 = ((omega**2 / c2_sq - k3_sq) * ((0.0_rk, 1.0_rk) * k3_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k3_val * R) / R**3)
    term2_n4 = ((omega**2 / c2_sq - k4_sq) * ((0.0_rk, 1.0_rk) * k4_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k4_val * R) / R**3)
    term2 = term2_prefactor * (B2 * term2_n2 + B3 * term2_n3 + B4 * term2_n4) * identity(3)

    ! -I_3\frac{\rho j\nu^2}{\pi(\lambda_c+2\mu_c)(\mu+\nu)(\mu_c+\nu_c)} *
    ! \sum_{n=2,3,4}B_n \paren{ik_nr-1}\frac{e^{ik_nr}}{r^3}
    term3_prefactor = J * nu**2 / (pi * (lam_c + 2.0_rk * mu_c) * (mu + nu) * (mu_c + nu_c))
    term3_n2 = ((0.0_rk, 1.0_rk) * k2_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k2_val * R) / R**3
    term3_n3 = ((0.0_rk, 1.0_rk) * k3_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k3_val * R) / R**3
    term3_n4 = ((0.0_rk, 1.0_rk) * k4_val * R - 1.0_rk) * exp((0.0_rk, 1.0_rk) * k4_val * R) / R**3
    term3 = term3_prefactor * (B2 * term3_n2 + B3 * term3_n3 + B4 * term3_n4) * identity(3)

    ! - \hat{r}\hat{r}^T \frac{\rho j(\lambda_c+\mu_c-\nu_c)}{4\pi(\lambda_c+2\mu_c)(\mu_c+\nu_c)} *
    ! \sum_{n=2,3,4} B_n  \paren{\frac{\omega^2}{c_2^2}-k_n^2}\paren{-k_n^2r^2-3ik_nr+3} \frac{e^{ik_nr}}{r^3}
    term4_prefactor = -J * (lam_c + mu_c - nu_c) / (4.0_rk * pi * (lam_c + 2.0_rk * mu_c) * (mu_c + nu_c))
    term4_n2 = ((omega**2 / c2_sq - k2_sq) * (-k2_sq * R**2 - (0.0_rk, 3.0_rk) * k2_val * R + 3) &
      * exp((0.0_rk, 1.0_rk) * k2_val * R) / R**3)
    term4_n3 = ((omega**2 / c2_sq - k3_sq) * (-k3_sq * R**2 - (0.0_rk, 3.0_rk) * k3_val * R + 3) &
      * exp((0.0_rk, 1.0_rk) * k3_val * R) / R**3)
    term4_n4 = ((omega**2 / c2_sq - k4_sq) * (-k4_sq * R**2 - (0.0_rk, 3.0_rk) * k4_val * R + 3) &
      * exp((0.0_rk, 1.0_rk) * k4_val * R) / R**3)
    term4 = term4_prefactor * (B2 * term4_n2 + B3 * term4_n3 + B4 * term4_n4) * outer_product(R_hat, R_hat)

    ! \hat{r}\hat{r}^T \frac{\rho j\nu^2}{\pi(\lambda_c+2\mu_c)(\mu+\nu)(\mu_c+\nu_c)} *
    ! \sum_{n=2,3,4} B_n \paren{-k_n^2r^2-3ik_nr+3} \frac{e^{ik_nr}}{r^3}
    term5_prefactor = J * nu**2 / (pi * (lam_c + 2.0_rk * mu_c) * (mu + nu) * (mu_c + nu_c))
    term5_n2 = (-k2_sq * R**2 - (0.0_rk, 3.0_rk) * k2_val * R + 3) * exp((0.0_rk, 1.0_rk) * k2_val * R) / R**3
    term5_n3 = (-k3_sq * R**2 - (0.0_rk, 3.0_rk) * k3_val * R + 3) * exp((0.0_rk, 1.0_rk) * k3_val * R) / R**3
    term5_n4 = (-k4_sq * R**2 - (0.0_rk, 3.0_rk) * k4_val * R + 3) * exp((0.0_rk, 1.0_rk) * k4_val * R) / R**3
    term5 = term5_prefactor * (B2 * term5_n2 + B3 * term5_n3 + B4 * term5_n4) * outer_product(R_hat, R_hat)

    G = 0.0_rk
    G(1:3, 1:3) = displacement_term
    G(4:6, 1:3) = term1 + term2 + term3 + term4 + term5
  end function greens_rotation_force

  function greens_rotation_force_static(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(G)
    real(rk), intent(in) :: x(3)
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: G(6, 3)

    real(rk) :: R
    real(rk) :: R_hat(3)

    R = sqrt(sum(x**2))
    R_hat = x / R

    ! TODO: derive the correct static Green's function. For now, return zeros
    G = 0.0_rk
  end function greens_rotation_force_static
end module cosserat_core
