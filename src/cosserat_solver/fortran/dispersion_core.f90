module dispersion_core
  use cosserat_branch_consts
  use cosserat_kinds, only: rk
  implicit none
  private
  public :: dispersion_A, dispersion_B, dispersion_C, dispersion, dispersion_zero
  public :: init_dispersion, c_pm, c_pm_prime
  !--- Parameters ---
  real(rk) :: rho, lam, mu, nu, J, lam_c, mu_c, nu_c

contains

  subroutine print_kind_info()
    use iso_fortran_env, only: output_unit
    implicit none

    write(output_unit, *) "rk kind        =", rk
    write(output_unit, *) "rk precision   =", precision(0.0_rk)
    write(output_unit, *) "rk range       =", range(0.0_rk)
    write(output_unit, *) "real(rk) bytes =", storage_size(0.0_rk)/8
    write(output_unit, *) "complex(rk) bytes =", storage_size((0.0_rk,0.0_rk))/8

    flush(output_unit)
  end subroutine print_kind_info

  !---------------------------------------------
  ! Initialize the material parameters
  !---------------------------------------------
  subroutine init_dispersion(rho_in, lam_in, mu_in, nu_in, J_in, lam_c_in, mu_c_in, nu_c_in)
    real(rk), intent(in) :: rho_in, lam_in, mu_in, nu_in, J_in, lam_c_in, mu_c_in, nu_c_in
    ! call print_kind_info() ! Uncomment for debugging kind info
    rho = rho_in
    lam = lam_in
    mu  = mu_in
    nu  = nu_in
    J   = J_in
    lam_c = lam_c_in
    mu_c  = mu_c_in
    nu_c  = nu_c_in
  end subroutine init_dispersion

  !---------------------------------------------
  ! Dispersion coefficients
  !---------------------------------------------
  function dispersion_A(r) result(A)
    complex(rk), intent(in) :: r
    complex(rk) :: A

    A = (0.0_rk, 2.0_rk) * nu / sqrt(rho * J)
  end function dispersion_A

  function dispersion_B(r) result(B)
    complex(rk), intent(in) :: r
    complex(rk) :: B
    real(rk) :: diff

    diff = (mu + nu)/rho - (mu_c + nu_c)/J
    B = r**2 * diff - 4.0_rk * nu / J
  end function dispersion_B

  function dispersion_C(r) result(C)
    complex(rk), intent(in) :: r
    complex(rk) :: C

    C = (0.0_rk, 2.0_rk) * nu * r**2 / sqrt(rho * J)
    ! print *, "dispersion_C debug: FORTRAN"
    ! print *, r, r**2, sqrt(rho * J), nu, (0.0_rk, 2.0_rk) * nu, C
  end function dispersion_C

  !---------------------------------------------
  ! Dispersion relation left-hand side
  !---------------------------------------------

  function dispersion(r, c) result(lhs)
    complex(rk), intent(in) :: r, c
    complex(rk) :: lhs
    complex(rk), volatile :: nonconstant
    complex(rk) :: A, B, C_coeff

    A = dispersion_A(r)
    B = dispersion_B(r)
    C_coeff = dispersion_C(r)

    nonconstant = A * c + B
    lhs = c * nonconstant + C_coeff
  end function dispersion

  function dispersion_zero(r, branch) result(lhs)
    complex(rk), intent(in) :: r
    integer, intent(in) :: branch
    complex(rk) :: lhs
    complex(rk), volatile :: nonconstant
    complex(rk) :: c

    c = c_pm(r, branch)

    nonconstant = dispersion_A(r) * c + dispersion_B(r)
    lhs = c * nonconstant + dispersion_C(r)
  end function dispersion_zero

  !---------------------------------------------
  ! Solve for c_pm
  !---------------------------------------------
function c_pm(r, branch) result(c)
  complex(rk), intent(in) :: r
  integer, intent(in) :: branch
  complex(rk) :: c
  complex(rk) :: A, B, C_coeff, sqrt_term, noncancelling
  real(rk) :: B_re

  ! compute coefficients
  A = dispersion_A(r)
  B = dispersion_B(r)
  C_coeff = dispersion_C(r)

  ! get real part of B
  B_re = real(B)

  ! sqrt term
  sqrt_term = sqrt(B**2 - 4.0_rk*A*C_coeff)

  ! choose root according to branch
  if (B_re < 0.0_rk) then
    noncancelling = (-B + sqrt_term) / (2.0_rk*A)
    if (branch < 0) then
      c = r**2 / noncancelling
    else
      c = noncancelling
    end if
  else
    noncancelling = (-B - sqrt_term) / (2.0_rk*A)
    if (branch < 0) then
      c = r**2 / noncancelling
    else
      c = noncancelling
    end if
  end if

  ! print *, "A =", A
  ! print *, "B =", B
  ! print *, "C =", C_coeff
  ! print *, "FORTRAN c_\pm =", c

end function c_pm


  !---------------------------------------------
  ! Implicit derivative c_pm'
  !---------------------------------------------
  function c_pm_prime(r, branch) result(dcdk)
    complex(rk), intent(in) :: r
    integer, intent(in) :: branch
    complex(rk) :: dcdk
    complex(rk) :: cp
    real(rk) :: diff
    complex(rk) :: num, denom

    cp = c_pm(r, branch)
    diff = (mu + nu)/rho - (mu_c + nu_c)/J

    num = -2.0_rk * cp * r * diff - (0.0_rk,4.0_rk) * nu * r / sqrt(rho * J)
    denom = r**2 * diff - 4.0_rk * nu / J + (0.0_rk,4.0_rk) * nu * cp / sqrt(rho * J)
    dcdk = num / denom
  end function c_pm_prime

end module dispersion_core
