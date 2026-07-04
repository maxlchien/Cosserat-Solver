!=====================================================
! C-compatible wrapper module for Python interface
!=====================================================
module dispersion_core_c
  use iso_c_binding
  use cosserat_branch_consts
  use cosserat_kinds, only: rk
  use dispersion_core
  implicit none
contains

  !---------------------------------------------
  ! Wrapper for c_pm
  !---------------------------------------------
  subroutine c_pm_wrapper(r_real, r_imag, branch, rho, mu, nu, J, mu_c, nu_c, &
                          result_real, result_imag) &
                          bind(C, name="c_pm_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    integer(c_int), intent(in), value :: branch
    real(c_double), intent(in), value :: rho, mu, nu, J, mu_c, nu_c
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    real(rk) :: rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q
    integer :: branch_internal

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    branch_internal = int(branch)
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    ! Call the actual function
    result_quad = c_pm(r_quad, branch_internal, rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine c_pm_wrapper

  !---------------------------------------------
  ! Wrapper for c_pm_prime
  !---------------------------------------------
  subroutine c_pm_prime_wrapper(r_real, r_imag, branch, rho, mu, nu, J, mu_c, nu_c, &
                                 result_real, result_imag) &
                                 bind(C, name="c_pm_prime_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    integer(c_int), intent(in), value :: branch
    real(c_double), intent(in), value :: rho, mu, nu, J, mu_c, nu_c
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    real(rk) :: rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q
    integer :: branch_internal

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    branch_internal = int(branch)
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    ! Call the actual function
    result_quad = c_pm_prime(r_quad, branch_internal, rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine c_pm_prime_wrapper


  !---------------------------------------------
  ! Wrapper for dispersion_A
  !---------------------------------------------
  subroutine dispersion_A_wrapper(r_real, r_imag, rho, nu, J, &
                                  result_real, result_imag) &
                                  bind(C, name="dispersion_A_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    real(c_double), intent(in), value :: rho, nu, J
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    real(rk) :: rho_q, nu_q, J_q

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    rho_q = real(rho, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)

    ! Call the actual function
    result_quad = dispersion_A(r_quad, rho_q, nu_q, J_q)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_A_wrapper

  !---------------------------------------------
  ! Wrapper for dispersion_B
  !---------------------------------------------
  subroutine dispersion_B_wrapper(r_real, r_imag, rho, mu, nu, J, mu_c, nu_c, &
                                  result_real, result_imag) &
                                  bind(C, name="dispersion_B_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    real(c_double), intent(in), value :: rho, mu, nu, J, mu_c, nu_c
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    real(rk) :: rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    ! Call the actual function
    result_quad = dispersion_B(r_quad, rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_B_wrapper

  !---------------------------------------------
  ! Wrapper for dispersion_C
  !---------------------------------------------
  subroutine dispersion_C_wrapper(r_real, r_imag, rho, nu, J, &
                                  result_real, result_imag) &
                                  bind(C, name="dispersion_C_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    real(c_double), intent(in), value :: rho, nu, J
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    real(rk) :: rho_q, nu_q, J_q

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    rho_q = real(rho, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)

    ! Call the actual function
    result_quad = dispersion_C(r_quad, rho_q, nu_q, J_q)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_C_wrapper

  subroutine dispersion_wrapper(r_real, r_imag, c_real, c_imag, &
                                rho, mu, nu, J, mu_c, nu_c, &
                                result_real, result_imag) &
                                bind(C, name="dispersion_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    real(c_double), intent(in), value :: c_real, c_imag
    real(c_double), intent(in), value :: rho, mu, nu, J, mu_c, nu_c
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, c_quad, result_quad
    real(rk) :: rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    c_quad = cmplx(real(c_real, kind=rk), real(c_imag, kind=rk), kind=rk)
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    ! Call the actual function
    result_quad = dispersion(r_quad, c_quad, rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_wrapper

  subroutine dispersion_zero_wrapper(r_real, r_imag, branch, &
                                rho, mu, nu, J, mu_c, nu_c, &
                                result_real, result_imag) &
                                bind(C, name="dispersion_zero_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    integer, intent(in), value :: branch
    real(c_double), intent(in), value :: rho, mu, nu, J, mu_c, nu_c
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    real(rk) :: rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    ! Call the actual function
    result_quad = dispersion_zero(r_quad, branch, rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_zero_wrapper
end module dispersion_core_c
