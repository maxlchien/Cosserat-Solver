module dispersion_core
  use cosserat_branch_consts
  use cosserat_kinds, only: rk
  implicit none
  private
  public :: dispersion_A, dispersion_B, dispersion_C, dispersion, dispersion_zero
  public :: c_pm, c_pm_prime

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
  ! Dispersion coefficients
  !---------------------------------------------
  function dispersion_A(r, rho, nu, J) result(A)
    complex(rk), intent(in) :: r
    real(rk), intent(in) :: rho, nu, J
    complex(rk) :: A

    A = (0.0_rk, 2.0_rk) * nu / sqrt(rho * J)
  end function dispersion_A

  function dispersion_B(r, rho, mu, nu, J, mu_c, nu_c) result(B)
    complex(rk), intent(in) :: r
    real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c
    complex(rk) :: B
    real(rk) :: diff

    diff = (mu + nu)/rho - (mu_c + nu_c)/J
    B = r**2 * diff - 4.0_rk * nu / J
  end function dispersion_B

  function dispersion_C(r, rho, nu, J) result(C)
    complex(rk), intent(in) :: r
    real(rk), intent(in) :: rho, nu, J
    complex(rk) :: C

    C = (0.0_rk, 2.0_rk) * nu * r**2 / sqrt(rho * J)
    ! print *, "dispersion_C debug: FORTRAN"
    ! print *, r, r**2, sqrt(rho * J), nu, (0.0_rk, 2.0_rk) * nu, C
  end function dispersion_C

  !---------------------------------------------
  ! Dispersion relation left-hand side
  !---------------------------------------------

  function dispersion(r, c, rho, mu, nu, J, mu_c, nu_c) result(lhs)
    complex(rk), intent(in) :: r, c
    real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c
    complex(rk) :: lhs
    complex(rk), volatile :: nonconstant
    complex(rk) :: A, B, C_coeff

    A = dispersion_A(r, rho, nu, J)
    B = dispersion_B(r, rho, mu, nu, J, mu_c, nu_c)
    C_coeff = dispersion_C(r, rho, nu, J)

    nonconstant = A * c + B
    lhs = c * nonconstant + C_coeff
  end function dispersion

  function dispersion_zero(r, branch, rho, mu, nu, J, mu_c, nu_c) result(lhs)
    complex(rk), intent(in) :: r
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c
    complex(rk) :: lhs
    complex(rk), volatile :: nonconstant
    complex(rk) :: c

    c = c_pm(r, branch, rho, mu, nu, J, mu_c, nu_c)

    nonconstant = dispersion_A(r, rho, nu, J) * c + dispersion_B(r, rho, mu, nu, J, mu_c, nu_c)
    lhs = c * nonconstant + dispersion_C(r, rho, nu, J)
  end function dispersion_zero

  !---------------------------------------------
  ! Solve for c_pm
  !---------------------------------------------
function c_pm(r, branch, rho, mu, nu, J, mu_c, nu_c) result(c)
  complex(rk), intent(in) :: r
  integer, intent(in) :: branch
  real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c
  complex(rk) :: c
  complex(rk) :: A, B, C_coeff, sqrt_term, noncancelling
  real(rk) :: B_re

  ! compute coefficients
  A = dispersion_A(r, rho, nu, J)
  B = dispersion_B(r, rho, mu, nu, J, mu_c, nu_c)
  C_coeff = dispersion_C(r, rho, nu, J)

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
  function c_pm_prime(r, branch, rho, mu, nu, J, mu_c, nu_c) result(dcdk)
    complex(rk), intent(in) :: r
    integer, intent(in) :: branch
    real(rk), intent(in) :: rho, mu, nu, J, mu_c, nu_c
    complex(rk) :: dcdk
    complex(rk) :: cp
    real(rk) :: diff
    complex(rk) :: num, denom

    cp = c_pm(r, branch, rho, mu, nu, J, mu_c, nu_c)
    diff = (mu + nu)/rho - (mu_c + nu_c)/J

    num = -2.0_rk * cp * r * diff - (0.0_rk,4.0_rk) * nu * r / sqrt(rho * J)
    denom = r**2 * diff - 4.0_rk * nu / J + (0.0_rk,4.0_rk) * nu * cp / sqrt(rho * J)
    dcdk = num / denom
  end function c_pm_prime

end module dispersion_core
