module integrator_core_c
  use iso_c_binding
  use cosserat_kinds, only: rk
  use integrator_core
  implicit none
contains

  !-----------------------------------------
  ! Initialization
  !-----------------------------------------
  subroutine integrator_init_c( &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c) &
      bind(C, name="integrator_init")
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c

    call init_integrator( &
        real(rho, rk), real(lam, rk), real(mu, rk), real(nu, rk), &
        real(J, rk), real(lam_c, rk), real(mu_c, rk), real(nu_c, rk))
  end subroutine integrator_init_c

    subroutine denom_c( &
      r_re, r_im, omega_re, omega_im, branch, &
      out_re, out_im) &
      bind(C, name="denom")
    real(c_double), value :: r_re, r_im
    real(c_double), value :: omega_re, omega_im
    integer(c_int), value :: branch
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: r, omega, val

    r     = cmplx(r_re, r_im, kind=rk)
    omega = cmplx(omega_re, omega_im, kind=rk)

    val = denom(r, omega, branch)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine denom_c

    subroutine denom_prime_c( &
      r_re, r_im, omega_re, omega_im, branch, &
      out_re, out_im) &
      bind(C, name="denom_prime")
    real(c_double), value :: r_re, r_im
    real(c_double), value :: omega_re, omega_im
    integer(c_int), value :: branch
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: r, omega, val

    r     = cmplx(r_re, r_im, kind=rk)
    omega = cmplx(omega_re, omega_im, kind=rk)

    val = denom_prime(r, omega, branch)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine denom_prime_c


    subroutine get_r2_poles_and_branches_c( &
      omega_re, omega_im, &
      r2_re, r2_im, branch, n) &
      bind(C, name="get_r2_poles_and_branches")
    real(c_double), value :: omega_re, omega_im
    real(c_double), intent(out) :: r2_re(2), r2_im(2)
    integer(c_int), intent(out) :: branch(2)
    integer(c_int), intent(out) :: n

    complex(rk) :: omega
    complex(rk) :: r2(2)
    integer :: br(2)
    integer :: i

    omega = cmplx(omega_re, omega_im, kind=rk)

    call get_r2_poles_and_branches(omega, r2, br, n)

    do i = 1, n
      r2_re(i) = real(r2(i), rk)
      r2_im(i) = aimag(r2(i))
      branch(i) = br(i)
    end do
  end subroutine get_r2_poles_and_branches_c

    subroutine pick_pole_c( &
      r2_re, r2_im, omega_re, omega_im, &
      out_re, out_im) &
      bind(C, name="pick_pole")
    real(c_double), value :: r2_re, r2_im
    real(c_double), value :: omega_re, omega_im
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: r2, omega, r

    r2    = cmplx(r2_re, r2_im, kind=rk)
    omega = cmplx(omega_re, omega_im, kind=rk)

    r = pick_pole(r2, omega)

    out_re = real(r, rk)
    out_im = aimag(r)
  end subroutine pick_pole_c



  !-----------------------------------------
  ! Integral 3_0
  !-----------------------------------------
  subroutine integral_3_0_c( &
      omega_re, omega_im, normx_re, normx_im, branch, &
      out_re, out_im) &
      bind(C, name="integral_3_0")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: normx_re, normx_im
    integer(c_int), value :: branch
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: omega, normx, val

    omega = cmplx(omega_re, omega_im, kind=rk)
    normx = cmplx(normx_re, normx_im, kind=rk)

    val = integral_3_0(omega, normx, branch)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine integral_3_0_c


  !-----------------------------------------
  ! Integral 3_2
  !-----------------------------------------
  subroutine integral_3_2_c( &
      omega_re, omega_im, normx_re, normx_im, branch, &
      out_re, out_im) &
      bind(C, name="integral_3_2")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: normx_re, normx_im
    integer(c_int), value :: branch
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: omega, normx, val

    omega = cmplx(omega_re, omega_im, kind=rk)
    normx = cmplx(normx_re, normx_im, kind=rk)

    val = integral_3_2(omega, normx, branch)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine integral_3_2_c


  !-----------------------------------------
  ! Integral 2_1
  !-----------------------------------------
  subroutine integral_2_1_c( &
      omega_re, omega_im, normx_re, normx_im, branch, &
      out_re, out_im) &
      bind(C, name="integral_2_1")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: normx_re, normx_im
    integer(c_int), value :: branch
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: omega, normx, val

    omega = cmplx(omega_re, omega_im, kind=rk)
    normx = cmplx(normx_re, normx_im, kind=rk)

    val = integral_2_1(omega, normx, branch)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine integral_2_1_c


  !-----------------------------------------
  ! Integral 1_0
  !-----------------------------------------
  subroutine integral_1_0_c( &
      omega_re, omega_im, normx_re, normx_im, branch, &
      out_re, out_im) &
      bind(C, name="integral_1_0")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: normx_re, normx_im
    integer(c_int), value :: branch
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: omega, normx, val

    omega = cmplx(omega_re, omega_im, kind=rk)
    normx = cmplx(normx_re, normx_im, kind=rk)

    val = integral_1_0(omega, normx, branch)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine integral_1_0_c

end module integrator_core_c
