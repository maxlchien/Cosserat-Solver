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
  ! Wrapper for init_dispersion
  !---------------------------------------------
  subroutine init_dispersion_wrapper(rho_in, lam_in, mu_in, nu_in, &
                                      J_in, lam_c_in, mu_c_in, nu_c_in) &
                                      bind(C, name="init_dispersion_wrapper")
    real(c_double), intent(in), value :: rho_in, lam_in, mu_in, nu_in
    real(c_double), intent(in), value :: J_in, lam_c_in, mu_c_in, nu_c_in

    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    ! Convert to quad precision
    rho_q = real(rho_in, kind=rk)
    lam_q = real(lam_in, kind=rk)
    mu_q = real(mu_in, kind=rk)
    nu_q = real(nu_in, kind=rk)
    J_q = real(J_in, kind=rk)
    lam_c_q = real(lam_c_in, kind=rk)
    mu_c_q = real(mu_c_in, kind=rk)
    nu_c_q = real(nu_c_in, kind=rk)

    call init_dispersion(rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)
  end subroutine init_dispersion_wrapper

  !---------------------------------------------
  ! Wrapper for c_pm
  !---------------------------------------------
  subroutine c_pm_wrapper(r_real, r_imag, branch, result_real, result_imag) &
                          bind(C, name="c_pm_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    integer(c_int), intent(in), value :: branch
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    integer :: branch_internal

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    branch_internal = int(branch)

    ! Call the actual function
    result_quad = c_pm(r_quad, branch_internal)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine c_pm_wrapper

  !---------------------------------------------
  ! Wrapper for c_pm_prime (add your implementation)
  !---------------------------------------------
  subroutine c_pm_prime_wrapper(r_real, r_imag, branch, result_real, result_imag) &
                                 bind(C, name="c_pm_prime_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    integer(c_int), intent(in), value :: branch
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    integer :: branch_internal

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    branch_internal = int(branch)

    ! Call the actual function
    result_quad = c_pm_prime(r_quad, branch_internal)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine c_pm_prime_wrapper


  !---------------------------------------------
  ! Wrapper for dispersion_A
  !---------------------------------------------
  subroutine dispersion_A_wrapper(r_real, r_imag, result_real, result_imag) &
                                  bind(C, name="dispersion_A_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)

    ! Call the actual function
    result_quad = dispersion_A(r_quad)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_A_wrapper

  !---------------------------------------------
  ! Wrapper for dispersion_B
  !---------------------------------------------
  subroutine dispersion_B_wrapper(r_real, r_imag, result_real, result_imag) &
                                  bind(C, name="dispersion_B_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)

    ! Call the actual function
    result_quad = dispersion_B(r_quad)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_B_wrapper

  !---------------------------------------------
  ! Wrapper for dispersion_C
  !---------------------------------------------
  subroutine dispersion_C_wrapper(r_real, r_imag, result_real, result_imag) &
                                  bind(C, name="dispersion_C_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad

    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)

    ! Call the actual function
    result_quad = dispersion_C(r_quad)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_C_wrapper

  subroutine dispersion_wrapper(r_real, r_imag, c_real, c_imag, result_real, result_imag) &
                                bind(C, name="dispersion_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    real(c_double), intent(in), value :: c_real, c_imag
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, c_quad, result_quad
    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)
    c_quad = cmplx(real(c_real, kind=rk), real(c_imag, kind=rk), kind=rk)

    ! Call the actual function
    result_quad = dispersion(r_quad, c_quad)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_wrapper

  subroutine dispersion_zero_wrapper(r_real, r_imag, branch, result_real, result_imag) &
                                bind(C, name="dispersion_zero_wrapper")
    real(c_double), intent(in), value :: r_real, r_imag
    integer, intent(in), value :: branch
    real(c_double), intent(out) :: result_real, result_imag

    complex(rk) :: r_quad, result_quad
    ! Convert C double to Fortran quad
    r_quad = cmplx(real(r_real, kind=rk), real(r_imag, kind=rk), kind=rk)

    ! Call the actual function
    result_quad = dispersion_zero(r_quad, branch)

    ! Convert quad result back to C double
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)
  end subroutine dispersion_zero_wrapper
end module dispersion_core_c
