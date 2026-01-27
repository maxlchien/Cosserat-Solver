module integrator_core_c
  use iso_c_binding
  use cosserat_kinds, only: rk
  use integrator_core
  implicit none
contains

  !-----------------------------------------
  ! Denominator and derivative
  !-----------------------------------------
  subroutine denom_c( &
      r_re, r_im, omega_re, omega_im, branch, &
      rho, mu, nu, J, mu_c, nu_c, &
      out_re, out_im) &
      bind(C, name="denom")
    real(c_double), value :: r_re, r_im
    real(c_double), value :: omega_re, omega_im
    integer(c_int), value :: branch
    real(c_double), value :: rho, mu, nu, J, mu_c, nu_c
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: r, omega, val
    real(rk) :: rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q

    r     = cmplx(r_re, r_im, kind=rk)
    omega = cmplx(omega_re, omega_im, kind=rk)
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    val = denom(r, omega, branch, rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine denom_c

  subroutine denom_prime_c( &
      r_re, r_im, omega_re, omega_im, branch, &
      rho, mu, nu, J, mu_c, nu_c, &
      out_re, out_im) &
      bind(C, name="denom_prime")
    real(c_double), value :: r_re, r_im
    real(c_double), value :: omega_re, omega_im
    integer(c_int), value :: branch
    real(c_double), value :: rho, mu, nu, J, mu_c, nu_c
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: r, omega, val
    real(rk) :: rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q

    r     = cmplx(r_re, r_im, kind=rk)
    omega = cmplx(omega_re, omega_im, kind=rk)
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    val = denom_prime(r, omega, branch, rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine denom_prime_c

  !-----------------------------------------
  ! Poles and branches
  !-----------------------------------------
  subroutine get_r2_poles_and_branches_c( &
      omega_re, omega_im, &
      rho, mu, nu, J, mu_c, nu_c, &
      r2_re, r2_im, branch, n) &
      bind(C, name="get_r2_poles_and_branches")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: rho, mu, nu, J, mu_c, nu_c
    real(c_double), intent(out) :: r2_re(2), r2_im(2)
    integer(c_int), intent(out) :: branch(2)
    integer(c_int), intent(out) :: n

    complex(rk) :: omega
    complex(rk) :: r2(2)
    integer :: br(2)
    real(rk) :: rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q
    integer :: i

    omega = cmplx(omega_re, omega_im, kind=rk)
    rho_q = real(rho, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    call get_r2_poles_and_branches(omega, r2, br, n, rho_q, mu_q, nu_q, J_q, mu_c_q, nu_c_q)

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
  ! Integral wrappers
  !-----------------------------------------
  subroutine integral_3_0_c( &
      omega_re, omega_im, normx_re, normx_im, branch, &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &
      out_re, out_im) &
      bind(C, name="integral_3_0")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: normx_re, normx_im
    integer(c_int), value :: branch
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: omega, normx, val
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    omega = cmplx(omega_re, omega_im, kind=rk)
    normx = cmplx(normx_re, normx_im, kind=rk)
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    lam_c_q = real(lam_c, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    val = integral_3_0(normx, omega, branch, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine integral_3_0_c

  subroutine integral_3_2_c( &
      omega_re, omega_im, normx_re, normx_im, branch, &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &
      out_re, out_im) &
      bind(C, name="integral_3_2")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: normx_re, normx_im
    integer(c_int), value :: branch
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: omega, normx, val
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    omega = cmplx(omega_re, omega_im, kind=rk)
    normx = cmplx(normx_re, normx_im, kind=rk)
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    lam_c_q = real(lam_c, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    val = integral_3_2(normx, omega, branch, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine integral_3_2_c

  subroutine integral_2_1_c( &
      omega_re, omega_im, normx_re, normx_im, branch, &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &
      out_re, out_im) &
      bind(C, name="integral_2_1")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: normx_re, normx_im
    integer(c_int), value :: branch
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: omega, normx, val
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    omega = cmplx(omega_re, omega_im, kind=rk)
    normx = cmplx(normx_re, normx_im, kind=rk)
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    lam_c_q = real(lam_c, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    val = integral_2_1(normx, omega, branch, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine integral_2_1_c

  subroutine integral_1_0_c( &
      omega_re, omega_im, normx_re, normx_im, branch, &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &
      out_re, out_im) &
      bind(C, name="integral_1_0")
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: normx_re, normx_im
    integer(c_int), value :: branch
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(c_double), intent(out) :: out_re, out_im

    complex(rk) :: omega, normx, val
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    omega = cmplx(omega_re, omega_im, kind=rk)
    normx = cmplx(normx_re, normx_im, kind=rk)
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    lam_c_q = real(lam_c, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    val = integral_1_0(normx, omega, branch, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)

    out_re = real(val, rk)
    out_im = aimag(val)
  end subroutine integral_1_0_c

  !-----------------------------------------
  ! Green's function wrappers
  !-----------------------------------------
  subroutine greens_x_omega_P_c(x, omega_re, omega_im, &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &
      G) &
      bind(C, name="greens_x_omega_P")
    real(c_double), intent(in) :: x(2)
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(c_double), intent(out) :: G(3,3)

    complex(rk) :: omega
    complex(rk) :: G_loc(3,3)
    real(rk) :: x_fortran(2)
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_fortran = x
    omega = cmplx(omega_re, omega_im, kind=rk)
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    lam_c_q = real(lam_c, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    call greens_x_omega_P(x_fortran, omega, G_loc, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)

    G = G_loc
  end subroutine greens_x_omega_P_c

  subroutine greens_x_omega_plus_c(x, omega_re, omega_im, &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &
      G) &
      bind(C, name="greens_x_omega_plus")
    real(c_double), intent(in) :: x(2)
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(c_double), intent(out) :: G(3,3)

    complex(rk) :: omega
    complex(rk) :: G_loc(3,3)
    real(rk) :: x_fortran(2)
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_fortran = x
    omega = cmplx(omega_re, omega_im, kind=rk)
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    lam_c_q = real(lam_c, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    call greens_x_omega_plus(x_fortran, omega, G_loc, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)

    G = G_loc
  end subroutine greens_x_omega_plus_c

  subroutine greens_x_omega_minus_c(x, omega_re, omega_im, &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &
      G) &
      bind(C, name="greens_x_omega_minus")
    real(c_double), intent(in) :: x(2)
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(c_double), intent(out) :: G(3,3)

    complex(rk) :: omega
    complex(rk) :: G_loc(3,3)
    real(rk) :: x_fortran(2)
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_fortran = x
    omega = cmplx(omega_re, omega_im, kind=rk)
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    lam_c_q = real(lam_c, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    call greens_x_omega_minus(x_fortran, omega, G_loc, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)

    G = G_loc
  end subroutine greens_x_omega_minus_c

  subroutine greens_x_omega_c(x, omega_re, omega_im, &
      rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &
      G) &
      bind(C, name="greens_x_omega")
    real(c_double), intent(in) :: x(2)
    real(c_double), value :: omega_re, omega_im
    real(c_double), value :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(c_double), intent(out) :: G(3,3)

    complex(rk) :: omega
    complex(rk) :: G_loc(3,3)
    real(rk) :: x_fortran(2)
    real(rk) :: rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q

    x_fortran = x
    omega = cmplx(omega_re, omega_im, kind=rk)
    rho_q = real(rho, kind=rk)
    lam_q = real(lam, kind=rk)
    mu_q = real(mu, kind=rk)
    nu_q = real(nu, kind=rk)
    J_q = real(J, kind=rk)
    lam_c_q = real(lam_c, kind=rk)
    mu_c_q = real(mu_c, kind=rk)
    nu_c_q = real(nu_c, kind=rk)

    call greens_x_omega(x_fortran, omega, G_loc, rho_q, lam_q, mu_q, nu_q, J_q, lam_c_q, mu_c_q, nu_c_q)

    G = G_loc
  end subroutine greens_x_omega_c

end module integrator_core_c
