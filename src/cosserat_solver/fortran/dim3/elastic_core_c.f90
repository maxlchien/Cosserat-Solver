! C-compatible wrapper module for Python interface
!=====================================================
module elastic_core_c
  use iso_c_binding
  use cosserat_kinds, only: rk
  use elastic_core
  implicit none
contains

  !---------------------------------------------
  ! Wrapper for greens_displacement_force
  !---------------------------------------------
  subroutine greens_displacement_force_wrapper(x, omega, rho, lam, mu, result_real, result_imag) &
    bind(C, name="greens_displacement_force_wrapper")
    real(c_double), intent(in) :: x(3)
    real(c_double), intent(in) :: omega, rho, lam, mu
    real(c_double), intent(out) :: result_real(3, 3)
    real(c_double), intent(out) :: result_imag(3, 3)
    complex(rk) :: result_quad(3, 3)
    real(rk) :: x_quad(3)
    real(rk) :: omega_q, rho_q, lam_q, mu_q

    x_quad = real(x, rk)
    omega_q = real(omega, rk)
    rho_q = real(rho, rk)
    lam_q = real(lam, rk)
    mu_q = real(mu, rk)

    result_quad = greens_displacement_force(x_quad, omega_q, rho_q, lam_q, mu_q)
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine greens_displacement_force_wrapper
end module elastic_core_c
