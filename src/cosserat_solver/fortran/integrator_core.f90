module integrator_core
  use cosserat_branch_consts
  use cosserat_kinds, only: rk
  use dispersion_core
  implicit none
  private

  public :: denom, denom_prime
  public :: get_poles_and_branches, pick_pole, get_r2_poles_and_branches
  public :: integrate
  public :: integral_3_0, integral_3_2, integral_2_1, integral_1_0
  public :: rotation_matrix
  public :: greens_x_omega_P
  public :: greens_x_omega_plus
  public :: greens_x_omega_minus
  public :: greens_x_omega

  real(rk), parameter :: tol = 1e-12_rk

  real(rk), parameter :: pole_tol = 1.0e-6_rk
  real(rk), parameter :: pi = 3.141592653589793238462643383279502884197_rk

  ! Interface for AMOS ZBESH subroutine (F77 code)
  interface
    subroutine zbesh(zr, zi, fnu, kode, m, n, cyr, cyi, nz, ierr)
      double precision :: zr, zi, fnu
      integer :: kode, m, n
      double precision :: cyr(*)
      double precision :: cyi(*)
      integer :: nz, ierr
    end subroutine zbesh
  end interface

abstract interface
  function analytic_integrand(r, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(val)
    import rk
    complex(rk), intent(in) :: r
    complex(rk), intent(in) :: omega
    integer, intent(in) :: branch
    complex(rk), intent(in) :: normx
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: val
  end function analytic_integrand
end interface


contains

  !-------------------------
  ! Denominator and derivative
  !-------------------------
  function denom(r, omega, branch, rho, mu, nu, J, mu_c, nu_c) result(d)
    complex(rk), intent(in) :: r
    complex(rk), intent(in) :: omega
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c
    complex(rk) :: d, cpm
    cpm = c_pm(r, branch, rho, mu, nu, J, mu_c, nu_c)
    d = (mu+nu)*r**2 - rho*omega**2 + (0.0_rk,2.0_rk)*nu*sqrt(rho/J)*cpm
  end function denom

  function denom_prime(r, omega, branch, rho, mu, nu, J, mu_c, nu_c) result(d)
    complex(rk), intent(in) :: r
    complex(rk), intent(in) :: omega
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c
    complex(rk) :: d, cpm_prime
    cpm_prime = c_pm_prime(r, branch, rho, mu, nu, J, mu_c, nu_c)
    d = 2.0_rk*(mu+nu)*r + (0.0_rk,2.0_rk)*nu*sqrt(rho/J)*cpm_prime
  end function denom_prime

!-------------------------
! Accurate Hankel function of the first kind
! Uses scaled version for large arguments to avoid overflow
!-------------------------
function hankel1(order, z) result(h)
  use iso_fortran_env, only: real64
  integer, intent(in) :: order
  complex(rk), intent(in) :: z
  complex(rk) :: h
  real(real64) :: zr, zi, fnu, cyr(1), cyi(1), abs_z
  integer :: kode, m, n, nz, ierr
  real(rk), parameter :: scale_threshold = 10.0_rk

  ! AMOS ZBESH: Hankel function
  ! SUBROUTINE ZBESH(ZR, ZI, FNU, KODE, M, N, CYR, CYI, NZ, IERR)
  ! Note: zbesh expects double precision (real64), not real128
  zr = real(real(z, kind=rk), kind=real64)
  zi = real(aimag(z), kind=real64)
  fnu = real(order, kind=real64)
  abs_z = sqrt(zr**2 + zi**2)

  ! Use scaled version only for large arguments
  if (abs_z > real(scale_threshold, kind=real64)) then
    kode = 2  ! scaled: returns H(z) * exp(-i*z)
  else
    kode = 1  ! unscaled: returns H(z) directly
  end if

  m = 1     ! Hankel function of the first kind
  n = 1     ! compute only one function value

  call zbesh(zr, zi, fnu, kode, m, n, cyr, cyi, nz, ierr)

  if (ierr /= 0 .or. nz /= 0) then
    ! AMOS error or underflow
    h = cmplx(0.0_rk, 0.0_rk, kind=rk)
  else
    h = cmplx(real(cyr(1), kind=rk), real(cyi(1), kind=rk), kind=rk)
    ! If we used scaled version, multiply back exp(i*z) to get unscaled
    if (kode == 2) then
      h = h * exp((0.0_rk,1.0_rk) * z)
    end if
  end if
end function hankel1

  subroutine get_r2_poles_and_branches(omega, r2, branch, n, rho, mu, nu, J, mu_c, nu_c)
    complex(rk), intent(in) :: omega
    complex(rk), intent(out) :: r2(2)
    integer, intent(out) :: branch(2)
    integer, intent(out) :: n
    real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c

    real(rk) :: a4, a2, a0
    complex(rk) :: disc, pole1, pole2
    complex(rk) :: rtest
    integer :: i

    a4 = (mu + nu) * (mu_c + nu_c) / sqrt(rho * J)

    a2 = -sqrt(rho * J) * ( (mu + nu)/rho + (mu_c + nu_c)/J ) * omega**2 &
        + 4.0_rk * nu * mu / sqrt(rho * J)

    a0 = sqrt(rho * J) * omega**4 &
        - sqrt(rho * J) * (4.0_rk * nu / J) * omega**2

    disc = cmplx(a2*a2 - 4.0_rk*a4*a0, 0.0_rk, kind=rk)

    if (a2 < 0.0_rk) then
        pole1 = (-a2 + sqrt(disc)) / (2.0_rk * a4)
        pole2 = a0 / (a4 * pole1)
    else
        pole2 = (-a2 - sqrt(disc)) / (2.0_rk * a4)
        pole1 = a0 / (a4 * pole2)
    end if

    r2(1) = pole1
    r2(2) = pole2
    n = 2

    ! Determine branch for each pole
    do i = 1, n
        rtest = sqrt(r2(i))

        if (abs(denom(rtest, omega, PLUS_BRANCH, rho, mu, nu, J, mu_c, nu_c)) < pole_tol) then
            branch(i) = PLUS_BRANCH
        else if (abs(denom(rtest, omega, -PLUS_BRANCH, rho, mu, nu, J, mu_c, nu_c)) < pole_tol) then
            branch(i) = -PLUS_BRANCH
        else
            write(*,*) "Could not determine branch for pole r^2 =", r2(i)
            write(*,*) "Denom values:", denom(rtest, omega, PLUS_BRANCH, rho, mu, nu, J, mu_c, nu_c), &
                        denom(rtest, omega, -PLUS_BRANCH, rho, mu, nu, J, mu_c, nu_c)
            stop
        end if
    end do
  end subroutine get_r2_poles_and_branches


  !-------------------------
  ! Pick pole r from r^2
  !-------------------------
  function pick_pole(r2, omega) result(r)
    complex(rk), intent(in) :: r2
    complex(rk), intent(in) :: omega
    complex(rk) :: r

    if (abs(aimag(r2)) < 1.0e-15_rk .and. real(r2) > 0.0_rk) then
        if (real(omega) >= 0.0_rk) then
            r = cmplx(sqrt(real(r2)), 0.0_rk, kind=rk)
        else
            r = cmplx(-sqrt(real(r2)), 0.0_rk, kind=rk)
        end if
    else
        r = sqrt(r2)
        if (aimag(r) < 0.0_rk) r = -r
    end if
  end function pick_pole

  subroutine get_poles_and_branches(omega, r, branch, n, rho, mu, nu, J, mu_c, nu_c)
    complex(rk), intent(in) :: omega
    complex(rk), intent(out) :: r(2)
    integer, intent(out) :: branch(2)
    integer, intent(out) :: n
    real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c

    complex(rk) :: r2(2)
    integer :: r2_branch(2)
    integer :: i

    call get_r2_poles_and_branches(omega, r2, r2_branch, n, rho, mu, nu, J, mu_c, nu_c)

    do i = 1, n
        r(i) = pick_pole(r2(i), omega)
        branch(i) = r2_branch(i)
    end do
  end subroutine get_poles_and_branches


  !-------------------------
  ! Residue integration
  !-------------------------
  function integrate(analytic, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(val)
    procedure(analytic_integrand) :: analytic
    complex(rk), intent(in) :: omega, normx
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: val, r(2)
    integer :: b(2), n, i

    call get_poles_and_branches(omega, r, b, n, rho, mu, nu, J, mu_c, nu_c)
    val = (0.0_rk,0.0_rk)

    do i=1,n
      if (b(i) /= branch) cycle
      val = val + analytic(r(i), omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) / &
            denom_prime(r(i),omega,branch, rho, mu, nu, J, mu_c, nu_c)
    end do

    val = val * (0.0_rk,2.0_rk)*pi
  end function integrate


  !-------------------------
  ! Core Hankel integrals
  !-------------------------

  ! Integral 3_0
  function integral_3_0(normx, omega, branch, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(val)
    complex(rk), intent(in) :: normx, omega
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: val

    val = integrate(integrand_3_0_fcn, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) / (8.0_rk*pi)
  end function integral_3_0

  ! Integral 3_2
  function integral_3_2(normx, omega, branch, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(val)
    complex(rk), intent(in) :: normx, omega
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: val

    val = integrate(integrand_3_2_fcn, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) / (8.0_rk*pi)
  end function integral_3_2

  ! Integral 2_1
  function integral_2_1(normx, omega, branch, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(val)
    complex(rk), intent(in) :: normx, omega
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: val

    val = integrate(integrand_2_1_fcn, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) / (8.0_rk*pi) * sqrt(rho/J)
  end function integral_2_1

  ! Integral 1_0
  function integral_1_0(normx, omega, branch, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(val)
    complex(rk), intent(in) :: normx, omega
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: val

    val = integrate(integrand_1_0_fcn, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) / (8.0_rk*pi) * (rho/J)
  end function integral_1_0

  ! Integrand functions that take material parameters explicitly
  function integrand_3_0_fcn(r, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(v)
    complex(rk), intent(in) :: r, omega, normx
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: v
    complex(rk) :: cpm
    cpm = c_pm(r, branch, rho, mu, nu, J, mu_c, nu_c)
    v = (r**3 / (r**2 - cpm**2)) * hankel1(0, r*normx)
  end function integrand_3_0_fcn

  function integrand_3_2_fcn(r, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(v)
    complex(rk), intent(in) :: r, omega, normx
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: v
    complex(rk) :: cpm
    cpm = c_pm(r, branch, rho, mu, nu, J, mu_c, nu_c)
    v = (r**3 / (r**2 - cpm**2)) * hankel1(2, r*normx)
  end function integrand_3_2_fcn

  function integrand_2_1_fcn(r, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(v)
    complex(rk), intent(in) :: r, omega, normx
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: v
    complex(rk) :: cpm
    cpm = c_pm(r, branch, rho, mu, nu, J, mu_c, nu_c)
    v = (r**2 * cpm / (r**2 - cpm**2)) * hankel1(1, r*normx)
  end function integrand_2_1_fcn

  function integrand_1_0_fcn(r, omega, branch, normx, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(v)
    complex(rk), intent(in) :: r, omega, normx
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: v
    complex(rk) :: cpm
    cpm = c_pm(r, branch, rho, mu, nu, J, mu_c, nu_c)
    v = (-r * cpm**2 / (r**2 - cpm**2)) * hankel1(0, r*normx)
  end function integrand_1_0_fcn

! greens logic

  !-------------------------
  ! Rotation matrix
  !-------------------------
  subroutine rotation_matrix(phi, R)
    real(rk), intent(in) :: phi
    complex(rk), intent(out) :: R(3,3)
    real(rk) :: cos_phi, sin_phi

    cos_phi = cos(phi)
    sin_phi = sin(phi)

    R(1,1) = cmplx(cos_phi, 0.0_rk, kind=rk)
    R(1,2) = cmplx(-sin_phi, 0.0_rk, kind=rk)
    R(1,3) = cmplx(0.0_rk, 0.0_rk, kind=rk)

    R(2,1) = cmplx(sin_phi, 0.0_rk, kind=rk)
    R(2,2) = cmplx(cos_phi, 0.0_rk, kind=rk)
    R(2,3) = cmplx(0.0_rk, 0.0_rk, kind=rk)

    R(3,1) = cmplx(0.0_rk, 0.0_rk, kind=rk)
    R(3,2) = cmplx(0.0_rk, 0.0_rk, kind=rk)
    R(3,3) = cmplx(1.0_rk, 0.0_rk, kind=rk)
  end subroutine rotation_matrix

  !-------------------------
  ! Green's function for P branch
  !-------------------------
  subroutine greens_x_omega_P(x, omega, G, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    real(rk), intent(in) :: x(2)
    complex(rk), intent(in) :: omega
    complex(rk), intent(out) :: G(3,3)
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c

    real(rk) :: normx, phi, c_P
    complex(rk) :: unrotated(3,3), R(3,3), RT(3,3), temp(3,3)
    complex(rk) :: h0, h2, factor
    integer :: i

    normx = sqrt(x(1)**2 + x(2)**2)
    phi = atan2(x(2), x(1))

    c_P = sqrt((lam + 2.0_rk*mu) / rho)

    ! Compute Hankel functions
    h0 = hankel1(0, omega*normx/c_P)
    h2 = hankel1(2, omega*normx/c_P)

    ! Initialize unrotated matrix
    unrotated = cmplx(0.0_rk, 0.0_rk, kind=rk)
    unrotated(1,1) = h0 - h2
    unrotated(2,2) = h0 + h2

    factor = cmplx(0.0_rk, 1.0_rk, kind=rk) / (8.0_rk * (lam + 2.0_rk*mu))
    unrotated = unrotated * factor

    ! Get rotation matrix and transpose
    call rotation_matrix(phi, R)
    RT = transpose(R)

    ! Compute G = R * unrotated * R^T
    temp = matmul(R, unrotated)
    G = matmul(temp, RT)
  end subroutine greens_x_omega_P

  !-------------------------
  ! Green's function for + branch
  !-------------------------
  subroutine greens_x_omega_plus(x, omega, G, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    real(rk), intent(in) :: x(2)
    complex(rk), intent(in) :: omega
    complex(rk), intent(out) :: G(3,3)
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c

    real(rk) :: normx, phi
    complex(rk) :: unrotated(3,3), R(3,3), RT(3,3), temp(3,3)
    complex(rk) :: normx_c, i_unit

    normx = sqrt(x(1)**2 + x(2)**2)
    phi = atan2(x(2), x(1))
    normx_c = cmplx(normx, 0.0_rk, kind=rk)
    i_unit = cmplx(0.0_rk, 1.0_rk, kind=rk)

    ! Initialize unrotated matrix
    unrotated = cmplx(0.0_rk, 0.0_rk, kind=rk)

    unrotated(1,1) = integral_3_0(normx_c, omega, PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) + &
                     integral_3_2(normx_c, omega, PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    unrotated(2,2) = integral_3_0(normx_c, omega, PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) - &
                     integral_3_2(normx_c, omega, PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    unrotated(2,3) = i_unit * integral_2_1(normx_c, omega, PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    unrotated(3,2) = -i_unit * integral_2_1(normx_c, omega, PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    unrotated(3,3) = integral_1_0(normx_c, omega, PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    ! Get rotation matrix and transpose
    call rotation_matrix(phi, R)
    RT = transpose(R)

    ! Compute G = R * unrotated * R^T
    temp = matmul(R, unrotated)
    G = matmul(temp, RT)
  end subroutine greens_x_omega_plus

  !-------------------------
  ! Green's function for - branch
  !-------------------------
  subroutine greens_x_omega_minus(x, omega, G, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    real(rk), intent(in) :: x(2)
    complex(rk), intent(in) :: omega
    complex(rk), intent(out) :: G(3,3)
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c

    real(rk) :: normx, phi
    complex(rk) :: unrotated(3,3), R(3,3), RT(3,3), temp(3,3)
    complex(rk) :: normx_c, i_unit

    normx = sqrt(x(1)**2 + x(2)**2)
    phi = atan2(x(2), x(1))
    normx_c = cmplx(normx, 0.0_rk, kind=rk)
    i_unit = cmplx(0.0_rk, 1.0_rk, kind=rk)

    ! Initialize unrotated matrix
    unrotated = cmplx(0.0_rk, 0.0_rk, kind=rk)

    unrotated(1,1) = integral_3_0(normx_c, omega, -PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) + &
                     integral_3_2(normx_c, omega, -PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    unrotated(2,2) = integral_3_0(normx_c, omega, -PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) - &
                     integral_3_2(normx_c, omega, -PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    unrotated(2,3) = i_unit * integral_2_1(normx_c, omega, -PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    unrotated(3,2) = -i_unit * integral_2_1(normx_c, omega, -PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    unrotated(3,3) = integral_1_0(normx_c, omega, -PLUS_BRANCH, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    ! Get rotation matrix and transpose
    call rotation_matrix(phi, R)
    RT = transpose(R)

    ! Compute G = R * unrotated * R^T
    temp = matmul(R, unrotated)
    G = matmul(temp, RT)
  end subroutine greens_x_omega_minus

  !-------------------------
  ! Complete Green's function (all branches)
  !-------------------------
  subroutine greens_x_omega(x, omega, G, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    real(rk), intent(in) :: x(2)
    complex(rk), intent(in) :: omega
    complex(rk), intent(out) :: G(3,3)
    real(rk), intent(in) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c

    complex(rk) :: G_P(3,3), G_plus(3,3), G_minus(3,3)

    ! Handle omega = 0 case
    if (abs(omega) < tol) then
      G = cmplx(0.0_rk, 0.0_rk, kind=rk)
      return
    end if

    call greens_x_omega_P(x, omega, G_P, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    call greens_x_omega_plus(x, omega, G_plus, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    call greens_x_omega_minus(x, omega, G_minus, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    G = G_P + G_plus + G_minus
  end subroutine greens_x_omega

end module integrator_core
