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
    real(c_double), intent(in), value :: omega, rho, lam, mu
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

  subroutine greens_displacement_force_vectorized_wrapper(x, omega, n_omega, rho, lam, mu, &
    force_use_openmp, force_no_openmp, result_real, result_imag) &
    bind(C, name="greens_displacement_force_vectorized_wrapper")
    real(c_double), intent(in) :: x(3)
    integer(c_int), intent(in), value :: n_omega
    real(c_double), intent(in) :: omega(n_omega)
    real(c_double), intent(in), value :: rho, lam, mu
    logical(c_bool), intent(in), value :: force_use_openmp, force_no_openmp
    real(c_double), intent(out) :: result_real(n_omega, 3, 3)
    real(c_double), intent(out) :: result_imag(n_omega, 3, 3)

    real(rk), allocatable :: omega_q(:)
    complex(rk), allocatable :: result_quad(:,:,:)
    real(rk) :: x_quad(3)
    real(rk) :: rho_q, lam_q, mu_q
    integer :: i
    logical :: force_use, force_no

    ! Allocate arrays
    allocate(omega_q(n_omega))
    allocate(result_quad(n_omega, 3, 3))

    ! Convert inputs to rk kind
    x_quad = real(x, rk)
    rho_q = real(rho, rk)
    lam_q = real(lam, rk)
    mu_q = real(mu, rk)
    force_use = force_use_openmp
    force_no = force_no_openmp
    omega_q = real(omega, rk)

    ! Call vectorized Fortran function with conditional optional arguments
    if (force_use) then
      result_quad = greens_displacement_force_vectorized(x_quad, omega_q, n_omega, &
                                      rho_q, lam_q, mu_q, force_use_openmp=.true.)
    else if (force_no) then
      result_quad = greens_displacement_force_vectorized(x_quad, omega_q, n_omega, &
                                      rho_q, lam_q, mu_q, force_no_openmp=.true.)
    else
      result_quad = greens_displacement_force_vectorized(x_quad, omega_q, n_omega, &
                                      rho_q, lam_q, mu_q)
    end if

    ! Convert results back to C-compatible arrays
    result_real = real(real(result_quad), kind=c_double)
    result_imag = real(aimag(result_quad), kind=c_double)

    ! Deallocate arrays
    deallocate(omega_q)
    deallocate(result_quad)
  end subroutine greens_displacement_force_vectorized_wrapper
end module elastic_core_c
