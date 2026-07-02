! C-compatible wrapper module for Python interface
!=====================================================
module cosserat_core_c
  use iso_c_binding
  use cosserat_kinds, only: rk
  use cosserat_core
  implicit none
contains

  !---------------------------------------------
  ! Wrapper for greens_mixed_force
  !---------------------------------------------
  subroutine greens_mixed_force_wrapper(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
    bind(C, name="greens_mixed_force_wrapper")
    real(c_double), intent(in) :: x(3)
    real(c_double), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: result_real(6, 6)
    real(c_double), intent(out) :: result_imag(6, 6)
    complex(rk) :: result_quad(6, 6)
    real(rk) :: x_quad(3)
    real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_quad = real(x, rk)
    omega_q = real(omega, rk)
    rho_q = real(rho, rk)
    lam_q = real(lam, rk)
    mu_q = real(mu, rk)
    nu_q = real(nu, rk)
    J_q = real(J, rk)
    lam_c_q = real(lam_c, rk)
    mu_c_q = real(mu_c, rk)
    nu_c_q = real(nu_c, rk)

    result_quad = greens_mixed_force(x_quad, omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine greens_mixed_force_wrapper

  !---------------------------------------------
  ! Wrapper for greens_displacement_force
  !---------------------------------------------
  subroutine greens_displacement_force_wrapper(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
    bind(C, name="greens_displacement_force_wrapper")
    real(c_double), intent(in) :: x(3)
    real(c_double), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: result_real(6, 3)
    real(c_double), intent(out) :: result_imag(6, 3)
    complex(rk) :: result_quad(6, 3)
    real(rk) :: x_quad(3)
    real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_quad = real(x, rk)
    omega_q = real(omega, rk)
    rho_q = real(rho, rk)
    lam_q = real(lam, rk)
    mu_q = real(mu, rk)
    nu_q = real(nu, rk)
    J_q = real(J, rk)
    lam_c_q = real(lam_c, rk)
    mu_c_q = real(mu_c, rk)
    nu_c_q = real(nu_c, rk)

    result_quad = greens_displacement_force(x_quad, omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine greens_displacement_force_wrapper

  !---------------------------------------------
  ! Wrapper for greens_rotation_force
  !---------------------------------------------
  subroutine greens_rotation_force_wrapper(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
    bind(C, name="greens_rotation_force_wrapper")
    real(c_double), intent(in) :: x(3)
    real(c_double), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: result_real(6, 3)
    real(c_double), intent(out) :: result_imag(6, 3)
    complex(rk) :: result_quad(6, 3)
    real(rk) :: x_quad(3)
    real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_quad = real(x, rk)
    omega_q = real(omega, rk)
    rho_q = real(rho, rk)
    lam_q = real(lam, rk)
    mu_q = real(mu, rk)
    nu_q = real(nu, rk)
    J_q = real(J, rk)
    lam_c_q = real(lam_c, rk)
    mu_c_q = real(mu_c, rk)
    nu_c_q = real(nu_c, rk)

    result_quad = greens_rotation_force(x_quad, omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine greens_rotation_force_wrapper

  !---------------------------------------------
  ! Wrapper for greens_displacement_force_static
  !---------------------------------------------
  subroutine greens_displacement_force_static_wrapper(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
    bind(C, name="greens_displacement_force_static_wrapper")
    real(c_double), intent(in) :: x(3)
    real(c_double), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: result_real(6, 3)
    real(c_double), intent(out) :: result_imag(6, 3)
    complex(rk) :: result_quad(6, 3)
    real(rk) :: x_quad(3)
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_quad = real(x, rk)
    rho_q = real(rho, rk)
    lam_q = real(lam, rk)
    mu_q = real(mu, rk)
    nu_q = real(nu, rk)
    J_q = real(J, rk)
    lam_c_q = real(lam_c, rk)
    mu_c_q = real(mu_c, rk)
    nu_c_q = real(nu_c, rk)

    result_quad = greens_displacement_force_static(x_quad, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine greens_displacement_force_static_wrapper

  !---------------------------------------------
  ! Wrapper for greens_rotation_force_static
  !---------------------------------------------
  subroutine greens_rotation_force_static_wrapper(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
    bind(C, name="greens_rotation_force_static_wrapper")
    real(c_double), intent(in) :: x(3)
    real(c_double), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: result_real(6, 3)
    real(c_double), intent(out) :: result_imag(6, 3)
    complex(rk) :: result_quad(6, 3)
    real(rk) :: x_quad(3)
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_quad = real(x, rk)
    rho_q = real(rho, rk)
    lam_q = real(lam, rk)
    mu_q = real(mu, rk)
    nu_q = real(nu, rk)
    J_q = real(J, rk)
    lam_c_q = real(lam_c, rk)
    mu_c_q = real(mu_c, rk)
    nu_c_q = real(nu_c, rk)

    result_quad = greens_rotation_force_static(x_quad, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine greens_rotation_force_static_wrapper

end module cosserat_core_c
