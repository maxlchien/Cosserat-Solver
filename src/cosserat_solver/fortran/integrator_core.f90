module integrator_core
  use cosserat_branch_consts
  use cosserat_kinds, only: rk
  use dispersion_core
  implicit none
  private

  public :: init_integrator
  public :: denom, denom_prime
  public :: get_poles_and_branches, pick_pole
  public :: integrate
  public :: integral_3_0, integral_3_2, integral_2_1, integral_1_0

  real(rk) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c
  real(rk), parameter :: tol = 1e-12_rk

  real(rk), parameter :: pole_tol = 1.0e-6_rk
  real(rk), parameter :: pi = 3.141592653589793238462643383279502884197_rk


abstract interface
  function analytic_integrand(r, omega, branch) result(val)
    import rk
    complex(rk), intent(in) :: r
    complex(rk), intent(in) :: omega
    integer, intent(in) :: branch
    complex(rk) :: val
  end function analytic_integrand
end interface


contains

  subroutine init_integrator(rho_in, lam_in, mu_in, nu_in, J_in, lam_c_in, mu_c_in, nu_c_in)
    real(rk), intent(in) :: rho_in, lam_in, mu_in, nu_in, J_in, lam_c_in, mu_c_in, nu_c_in
    rho = rho_in
    lam = lam_in
    mu = mu_in
    nu = nu_in
    J = J_in
    lam_c = lam_c_in
    mu_c = mu_c_in
    nu_c = nu_c_in

    call init_dispersion(rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
  end subroutine init_integrator

  !-------------------------
  ! Denominator and derivative
  !-------------------------
  function denom(r, omega, branch) result(d)
    complex(rk), intent(in) :: r
    complex(rk), intent(in) :: omega
    integer, intent(in) :: branch
    complex(rk) :: d, cpm
    cpm = c_pm(r, branch)
    d = (mu+nu)*r**2 - rho*omega**2 + (0.0_rk,2.0_rk)*nu*sqrt(rho/J)*cpm
  end function denom

  function denom_prime(r, omega, branch) result(d)
    complex(rk), intent(in) :: r
    complex(rk), intent(in) :: omega
    integer, intent(in) :: branch
    complex(rk) :: d, cpm_prime
    cpm_prime = c_pm_prime(r, branch)
    d = 2.0_rk*(mu+nu)*r + (0.0_rk,2.0_rk)*nu*sqrt(rho/J)*cpm_prime
  end function denom_prime

!-------------------------
! Accurate Hankel function of the first kind
! Scaled via KODE=2, then multiply back exp(i*z)
!-------------------------
function hankel1(order, z) result(h)
  integer, intent(in) :: order
  complex(rk), intent(in) :: z
  complex(rk) :: h, hdummy
  integer :: kode

  ! AMOS ZBESH: scaled Hankel H_n^(1)(z)/exp(i*z)
  kode = 2
  call zbesh(order, z, h, hdummy, kode)

  ! Multiply back exp(i*z) to recover true Hankel
  h = h * cmplx(cos(real(z)), sin(real(z)), kind=rk) * exp(-aimag(z))
end function hankel1

  subroutine get_r2_poles_and_branches(omega, r2, branch, n)
    complex(rk), intent(in) :: omega
    complex(rk), intent(out) :: r2(2)
    integer, intent(out) :: branch(2)
    integer, intent(out) :: n

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

        if (abs(denom(rtest, omega, PLUS_BRANCH)) < pole_tol) then
            branch(i) = PLUS_BRANCH
        else if (abs(denom(rtest, omega, -PLUS_BRANCH)) < pole_tol) then
            branch(i) = -PLUS_BRANCH
        else
            write(*,*) "Could not determine branch for pole r^2 =", r2(i)
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

  subroutine get_poles_and_branches(omega, r, branch, n)
    complex(rk), intent(in) :: omega
    complex(rk), intent(out) :: r(2)
    integer, intent(out) :: branch(2)
    integer, intent(out) :: n

    complex(rk) :: r2(2)
    integer :: r2_branch(2)
    integer :: i

    call get_r2_poles_and_branches(omega, r2, r2_branch, n)

    do i = 1, n
        r(i) = pick_pole(r2(i), omega)
        branch(i) = r2_branch(i)
    end do
  end subroutine get_poles_and_branches




  !-------------------------
  ! Residue integration
  !-------------------------
  function integrate(analytic, omega, branch) result(val)
  procedure(analytic_integrand) :: analytic
  complex(rk), intent(in) :: omega
  integer, intent(in) :: branch
  complex(rk) :: val

  complex(rk) :: r(2), residue
  integer :: pole_branch(2), n, i

  call get_poles_and_branches(omega, r, pole_branch, n)

  val = cmplx(0.0_rk, 0.0_rk, kind=rk)

  do i = 1, n
      if (pole_branch(i) /= branch) cycle
      residue = analytic(r(i), omega, branch) &
                / denom_prime(r(i), omega, branch)
      val = val + residue
  end do

  val = val * cmplx(0.0_rk, 2.0_rk*pi, kind=rk)
end function integrate


  !-------------------------
  ! Core Hankel integrals
  !-------------------------
  ! Integral 3_0
function analytic_3_0(r, normx, branch) result(val)
  complex(rk), intent(in) :: r
  complex(rk), intent(in) :: normx
  integer, intent(in) :: branch
  complex(rk) :: val
  complex(rk) :: cpm

  cpm = c_pm(r, branch)
  val = (r**3 / (r**2 - cpm**2)) * hankel1(0, r*normx)
end function analytic_3_0

function integral_3_0(r, normx, branch) result(val)
  complex(rk), intent(in) :: r
  complex(rk), intent(in) :: normx
  integer, intent(in) :: branch
  complex(rk) :: val

  val = integrate(analytic_3_0, r, branch)
end function integral_3_0

! Integral 3_2
function analytic_3_2(r, normx, branch) result(val)
  complex(rk), intent(in) :: r
  complex(rk), intent(in) :: normx
  integer, intent(in) :: branch
  complex(rk) :: val
  complex(rk) :: cpm

  cpm = c_pm(r, branch)
  val = (r**3 / (r**2 - cpm**2)) * hankel1(2, r*normx)
end function analytic_3_2

function integral_3_2(r, normx, branch) result(val)
  complex(rk), intent(in) :: r
  complex(rk), intent(in) :: normx
  integer, intent(in) :: branch
  complex(rk) :: val

  val = integrate(analytic_3_2, r, branch)
end function integral_3_2

! Integral 2_1
function analytic_2_1(r, normx, branch) result(val)
  complex(rk), intent(in) :: r
  complex(rk), intent(in) :: normx
  integer, intent(in) :: branch
  complex(rk) :: val
  complex(rk) :: cpm

  cpm = c_pm(r, branch)
  val = (r**2 * cpm / (r**2 - cpm**2)) * hankel1(1, r*normx)
end function analytic_2_1

function integral_2_1(r, normx, branch) result(val)
  complex(rk), intent(in) :: r
  complex(rk), intent(in) :: normx
  integer, intent(in) :: branch
  complex(rk) :: val

  val = integrate(analytic_2_1, r, branch)
end function integral_2_1

! Integral 1_0
function analytic_1_0(r, normx, branch) result(val)
  complex(rk), intent(in) :: r
  complex(rk), intent(in) :: normx
  integer, intent(in) :: branch
  complex(rk) :: val
  complex(rk) :: cpm

  cpm = c_pm(r, branch)
  val = (-r * cpm**2 / (r**2 - cpm**2)) * hankel1(0, r*normx)
end function analytic_1_0

function integral_1_0(r, normx, branch) result(val)
  complex(rk), intent(in) :: r
  complex(rk), intent(in) :: normx
  integer, intent(in) :: branch
  complex(rk) :: val

  val = integrate(analytic_1_0, r, branch)
end function integral_1_0

end module integrator_core
