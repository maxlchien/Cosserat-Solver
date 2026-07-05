!=====================================================
! C-compatible wrapper module for Python interface
!=====================================================
module dispersion_core_c
  use iso_c_binding
  use cosserat_kinds, only: rk
  use dispersion_core
  implicit none
contains

  !---------------------------------------------
  ! Wrapper for c_n^2
  !---------------------------------------------
  subroutine c1_squared_wrapper(rho, lam, mu, result) &
                          bind(C, name="c1_squared_wrapper")
    real(c_double), intent(in), value :: rho, lam, mu
    real(c_double), intent(out) :: result

    real(rk) :: result_quad
    real(rk) :: rho_q, lam_q, mu_q

    ! Convert C double to Fortran quad
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)

    ! Call the actual function
    result_quad = c1_squared(rho_q, lam_q, mu_q)

    ! Convert quad result back to C double
    result = real(result_quad, kind=c_double)
  end subroutine c1_squared_wrapper

  subroutine c2_squared_wrapper(rho, mu, nu, result) &
                          bind(C, name="c2_squared_wrapper")
    real(c_double), intent(in), value :: rho, mu, nu
    real(c_double), intent(out) :: result

    real(rk) :: result_quad
    real(rk) :: rho_q, mu_q, nu_q

    ! Convert C double to Fortran quad
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)

    ! Call the actual function
    result_quad = c2_squared(rho_q, mu_q, nu_q)

    ! Convert quad result back to C double
    result = real(result_quad, kind=c_double)
  end subroutine c2_squared_wrapper

    subroutine c3_squared_wrapper(J, lam_c, mu_c, result) &
                            bind(C, name="c3_squared_wrapper")
        real(c_double), intent(in), value :: J, lam_c, mu_c
        real(c_double), intent(out) :: result

        real(rk) :: result_quad
        real(rk) :: J_q, lam_c_q, mu_c_q

        ! Convert C double to Fortran quad
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)

        ! Call the actual function
        result_quad = c3_squared(J_q, lam_c_q, mu_c_q)

        ! Convert quad result back to C double
        result = real(result_quad, kind=c_double)
    end subroutine c3_squared_wrapper

    subroutine c4_squared_wrapper(J, mu_c, nu_c, result) &
                            bind(C, name="c4_squared_wrapper")
        real(c_double), intent(in), value :: J, mu_c, nu_c
        real(c_double), intent(out) :: result

        real(rk) :: result_quad
        real(rk) :: J_q, mu_c_q, nu_c_q

        ! Convert C double to Fortran quad
        J_q = real(J, kind=rk)
        mu_c_q = real(mu_c, kind=rk)
        nu_c_q = real(nu_c, kind=rk)

        ! Call the actual function
        result_quad = c4_squared(J_q, mu_c_q, nu_c_q)

        ! Convert quad result back to C double
        result = real(result_quad, kind=c_double)
    end subroutine c4_squared_wrapper

  !---------------------------------------------
  ! Wrapper for w0_squared
  !---------------------------------------------

    subroutine w0_squared_wrapper(nu, J, result) &
                            bind(C, name="w0_squared_wrapper")
        real(c_double), intent(in), value :: nu, J
        real(c_double), intent(out) :: result

        real(rk) :: result_quad
        real(rk) :: nu_q, J_q

        ! Convert C double to Fortran quad
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)

        ! Call the actual function
        result_quad = w0_squared(nu_q, J_q)

        ! Convert quad result back to C double
        result = real(result_quad, kind=c_double)
    end subroutine w0_squared_wrapper

  !---------------------------------------------
  ! Wrapper for dispersion relation coefficients r, s
  !---------------------------------------------

    subroutine dispersion_r_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result) &
                            bind(C, name="dispersion_r_wrapper")
        real(c_double), intent(in), value :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        real(c_double), intent(out) :: result

        real(rk) :: result_quad
        real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        rho_q = real(rho, kind=rk)
        lam_q = real(lam, kind=rk)
        mu_q = real(mu, kind=rk)
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)
        nu_c_q = real(nu_c, kind=rk)

        ! Call the actual function
        result_quad = dispersion_r(omega_q, rho_q, lam_q, mu_q, nu_q, &
                                J_q, lam_c_q, mu_c_q, nu_c_q)

        ! Convert quad result back to C double
        result = real(result_quad, kind=c_double)
    end subroutine dispersion_r_wrapper

    subroutine dispersion_s_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result) &
                            bind(C, name="dispersion_s_wrapper")
        real(c_double), intent(in), value :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        real(c_double), intent(out) :: result

        real(rk) :: result_quad
        real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        rho_q = real(rho, kind=rk)
        lam_q = real(lam, kind=rk)
        mu_q = real(mu, kind=rk)
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)
        nu_c_q = real(nu_c, kind=rk)

        ! Call the actual function
        result_quad = dispersion_s(omega_q, rho_q, lam_q, mu_q, nu_q, &
                                   J_q, lam_c_q, mu_c_q, nu_c_q)

        ! Convert quad result back to C double
        result = real(result_quad, kind=c_double)
    end subroutine dispersion_s_wrapper


  !---------------------------------------------
  ! Wrapper for k_n^2
  !---------------------------------------------

    subroutine k1_squared_wrapper(omega, rho, lam, mu, result_real, result_imag) &
                            bind(C, name="k1_squared_wrapper")
        real(c_double), intent(in), value :: omega, rho, lam, mu
        real(c_double), intent(out) :: result_real, result_imag

        complex(rk) :: result_quad
        real(rk) :: omega_q, rho_q, lam_q, mu_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        rho_q = real(rho, kind=rk)
        lam_q = real(lam, kind=rk)
        mu_q = real(mu, kind=rk)

        ! Call the actual function
        result_quad = k1_squared(omega_q, rho_q, lam_q, mu_q)

        ! Convert quad result back to C double
        result_real = real(real(result_quad), kind=c_double)
        result_imag = real(aimag(result_quad), kind=c_double)
    end subroutine k1_squared_wrapper

    subroutine k2_squared_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
                            bind(C, name="k2_squared_wrapper")
        real(c_double), intent(in), value :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        real(c_double), intent(out) :: result_real, result_imag

        complex(rk) :: result_quad
        real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        rho_q = real(rho, kind=rk)
        lam_q = real(lam, kind=rk)
        mu_q = real(mu, kind=rk)
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)
        nu_c_q = real(nu_c, kind=rk)

        ! Call the actual function
        result_quad = k2_squared(omega_q, rho_q, lam_q, mu_q,  nu_q, &
                            J_q, lam_c_q, mu_c_q, nu_c_q)

        ! Convert quad result back to C double
        result_real = real(real(result_quad), kind=c_double)
        result_imag = real(aimag(result_quad), kind=c_double)
    end subroutine k2_squared_wrapper

    subroutine k3_squared_wrapper(omega, nu, J, lam_c, mu_c, result_real, result_imag) &
                            bind(C, name="k3_squared_wrapper")
        real(c_double), intent(in), value :: omega, nu, J, lam_c, mu_c
        real(c_double), intent(out) :: result_real, result_imag

        complex(rk) :: result_quad
        real(rk) :: omega_q, nu_q, J_q, lam_c_q, mu_c_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)

        ! Call the actual function
        result_quad = k3_squared(omega_q, nu_q, J_q, lam_c_q, mu_c_q)

        ! Convert quad result back to C double
        result_real = real(real(result_quad), kind=c_double)
        result_imag = real(aimag(result_quad), kind=c_double)
    end subroutine k3_squared_wrapper

    subroutine k4_squared_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
                            bind(C, name="k4_squared_wrapper")
        real(c_double), intent(in), value :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        real(c_double), intent(out) :: result_real, result_imag

        complex(rk) :: result_quad
        real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        rho_q = real(rho, kind=rk)
        lam_q = real(lam, kind=rk)
        mu_q = real(mu, kind=rk)
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)
        nu_c_q = real(nu_c, kind=rk)

        ! Call the actual function
        result_quad = k4_squared(omega_q, rho_q, lam_q, mu_q, nu_q, &
                                 J_q, lam_c_q, mu_c_q, nu_c_q)

        ! Convert quad result back to C double
        result_real = real(real(result_quad), kind=c_double)
        result_imag = real(aimag(result_quad), kind=c_double)
    end subroutine k4_squared_wrapper

  !---------------------------------------------
  ! Wrapper for k_n
  !---------------------------------------------

  subroutine k1_wrapper(omega, rho, lam, mu, result_real, result_imag) &
                            bind(C, name="k1_wrapper")
        real(c_double), intent(in), value :: omega, rho, lam, mu
        real(c_double), intent(out) :: result_real, result_imag

        complex(rk) :: result_quad
        real(rk) :: omega_q, rho_q, lam_q, mu_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        rho_q = real(rho, kind=rk)
        lam_q = real(lam, kind=rk)
        mu_q = real(mu, kind=rk)

        ! Call the actual function
        result_quad = k1(omega_q, rho_q, lam_q, mu_q)

        ! Convert quad result back to C double
        result_real = real(real(result_quad), kind=c_double)
        result_imag = real(aimag(result_quad), kind=c_double)
    end subroutine k1_wrapper

    subroutine k2_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
                            bind(C, name="k2_wrapper")
        real(c_double), intent(in), value :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        real(c_double), intent(out) :: result_real, result_imag

        complex(rk) :: result_quad
        real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        rho_q = real(rho, kind=rk)
        lam_q = real(lam, kind=rk)
        mu_q = real(mu, kind=rk)
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)
        nu_c_q = real(nu_c, kind=rk)

        ! Call the actual function
        result_quad = k2(omega_q, rho_q, lam_q, mu_q, nu_q, &
                            J_q, lam_c_q, mu_c_q, nu_c_q)

        ! Convert quad result back to C double
        result_real = real(real(result_quad), kind=c_double)
        result_imag = real(aimag(result_quad), kind=c_double)
    end subroutine k2_wrapper

    subroutine k3_wrapper(omega, nu, J, lam_c, mu_c, result_real, result_imag) &
                            bind(C, name="k3_wrapper")
        real(c_double), intent(in), value :: omega, nu, J, lam_c, mu_c
        real(c_double), intent(out) :: result_real, result_imag

        complex(rk) :: result_quad
        real(rk) :: omega_q, nu_q, J_q, lam_c_q, mu_c_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)

        ! Call the actual function
        result_quad = k3(omega_q, nu_q, J_q, lam_c_q, mu_c_q)

        ! Convert quad result back to C double
        result_real = real(real(result_quad), kind=c_double)
        result_imag = real(aimag(result_quad), kind=c_double)
    end subroutine k3_wrapper

    subroutine k4_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, result_real, result_imag) &
                            bind(C, name="k4_wrapper")
        real(c_double), intent(in), value :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        real(c_double), intent(out) :: result_real, result_imag

        complex(rk) :: result_quad
        real(rk) :: omega_q, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

        ! Convert C double to Fortran quad
        omega_q = real(omega, kind=rk)
        rho_q = real(rho, kind=rk)
        lam_q = real(lam, kind=rk)
        mu_q = real(mu, kind=rk)
        nu_q = real(nu, kind=rk)
        J_q = real(J, kind=rk)
        lam_c_q = real(lam_c, kind=rk)
        mu_c_q = real(mu_c, kind=rk)
        nu_c_q = real(nu_c, kind=rk)

        ! Call the actual function
        result_quad = k4(omega_q, rho_q, lam_q, mu_q, nu_q, &
                                 J_q, lam_c_q, mu_c_q, nu_c_q)

        ! Convert quad result back to C double
        result_real = real(real(result_quad), kind=c_double)
        result_imag = real(aimag(result_quad), kind=c_double)
    end subroutine k4_wrapper

end module dispersion_core_c
