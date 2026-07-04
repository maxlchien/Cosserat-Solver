module dispersion_core
  use cosserat_kinds, only: rk
  implicit none
  private
  public :: c1_squared, c2_squared, c3_squared, c4_squared
  public :: w0_squared
  public :: dispersion_r, dispersion_s
  public :: k1_squared, k2_squared, k3_squared, k4_squared
  public :: k1, k2, k3, k4

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
  ! Dispersion coefficients c_n^2
  !---------------------------------------------
  function c1_squared(rho, lam, mu) result(c1_sq)
    real(rk), intent(in) :: rho, lam, mu
    real(rk) :: c1_sq

    c1_sq = (lam + 2.0_rk * mu) / rho
  end function c1_squared

  function c2_squared(rho, mu, nu) result(c2_sq)
    real(rk), intent(in) :: rho, mu, nu
    real(rk) :: c2_sq

    c2_sq = (mu + nu) / rho
  end function c2_squared

  function c3_squared(J, lam_c, mu_c) result(c3_sq)
    real(rk), intent(in) :: J, lam_c, mu_c
    real(rk) :: c3_sq

    c3_sq = (lam_c + 2.0_rk * mu_c) / J
  end function c3_squared

  function c4_squared(J, mu_c, nu_c) result(c4_sq)
    real(rk), intent(in) :: J, mu_c, nu_c
    real(rk) :: c4_sq

    c4_sq = (mu_c + nu_c) / J
  end function c4_squared

  !---------------------------------------------
  ! Cutoff frequency
  !---------------------------------------------

  function w0_squared(nu, J) result(w0_sq)
    real(rk), intent(in) :: nu, J
    real(rk) :: w0_sq

    w0_sq = 4.0_rk * nu / J
  end function w0_squared

  !---------------------------------------------
  ! Coefficients r, s from dispersion relation for k2, k4
  ! See Eringen (1999) equation (5.11.20)
  !---------------------------------------------

  function dispersion_r(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(r)
    ! According to Eringen (1999) equation (5.11.20), this is computed by
    ! r = (1 + c_2^2 / c_4^2) * (omega^2 / c_2^2) * (1/2)
    !      - (1 - J * w_0^2 / (4 * rho * c_2^2)) * w_0^2 / (2 * c_4^2)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(rk) :: r
    real(rk) :: c2_sq, c4_sq, w0_sq
    real(rk) :: term1, term2

    c2_sq = c2_squared(rho, mu, nu)
    c4_sq = c4_squared(J, mu_c, nu_c)
    w0_sq = w0_squared(nu, J)
    term1 = (1.0_rk / 2.0_rk) * (1.0_rk + c2_sq / c4_sq) * omega**2 / c2_sq
    term2 = (1.0_rk - J * w0_sq / (4.0_rk * rho * c2_sq)) * w0_sq / (2.0_rk * c4_sq)

    r = term1 - term2
  end function dispersion_r

  function dispersion_s(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(s)
    ! According to Eringen (1999) equation (5.11.20), this is computed by
    ! s = (omega^2 / c_2^2) * (omega^2 / c_4^2 - omega_0^2 / c_4^2)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    real(rk) :: s
    real(rk) :: c2_sq, c4_sq, w0_sq
    real(rk) :: term1, term2

    c2_sq = c2_squared(rho, mu, nu)
    c4_sq = c4_squared(J, mu_c, nu_c)
    w0_sq = w0_squared(nu, J)

    s = (omega**2 / c2_sq) * (omega**2 / c4_sq - w0_sq / c4_sq)
  end function dispersion_s

  !---------------------------------------------
  ! Dispersion relation coefficients k_n^2
  !---------------------------------------------
  function k1_squared(omega, rho, lam, mu) result(k1_sq)
    real(rk), intent(in) :: omega, rho, lam, mu
    complex(rk) :: k1_sq

    k1_sq = omega**2 / c1_squared(rho, lam, mu)
  end function k1_squared

  function k2_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(k2_sq)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: k2_sq
    complex(rk) :: r, s

    r = dispersion_r(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    s = dispersion_s(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    k2_sq = r + sqrt(r**2 - s)
  end function k2_squared

  function k3_squared(omega, nu, J, lam_c, mu_c) result(k3_sq)
    real(rk), intent(in) :: omega, nu, J, lam_c, mu_c
    complex(rk) :: k3_sq

    k3_sq = (omega**2 - w0_squared(nu, J)) / c3_squared(J, lam_c, mu_c)
  end function k3_squared

  function k4_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(k4_sq)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: k4_sq
    complex(rk) :: r, s

    r = dispersion_r(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    s = dispersion_s(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    k4_sq = r - sqrt(r**2 - s)
  end function k4_squared

  !---------------------------------------------
  ! Dispersion relation coefficients k_n
  ! When k_n^2 >=0, the principal square root is multiplied by the sign of omega, because
  ! this is the correct analytic continuation for the elastic case.
  ! Otherwise, the principal square root is returned directly.
  ! More work needs to be done to verify that this is the correct analytic continuation.
  !---------------------------------------------

  function get_k_from_k_squared(omega, k_sq) result(k)
    real(rk), intent(in) :: omega
    complex(rk), intent(in) :: k_sq
    complex(rk) :: k

    if ((real(k_sq) >= 0.0_rk) .and. (aimag(k_sq) == 0.0_rk)) then
      k = sign(sqrt(real(k_sq)), omega)
    else
      k = sqrt(k_sq)
    end if
  end function get_k_from_k_squared


  function k1(omega, rho, lam, mu) result(k1_val)
    real(rk), intent(in) :: omega, rho, lam, mu
    complex(rk) :: k1_val
    complex(rk) :: k1_sq

    k1_sq = k1_squared(omega, rho, lam, mu)
    k1_val = get_k_from_k_squared(omega, k1_sq)
  end function k1

  function k2(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(k2_val)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: k2_val
    complex(rk) :: k2_sq

    k2_sq = k2_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k2_val = get_k_from_k_squared(omega, k2_sq)
  end function k2

  function k3(omega, nu, J, lam_c, mu_c) result(k3_val)
    real(rk), intent(in) :: omega, nu, J, lam_c, mu_c
    complex(rk) :: k3_val
    complex(rk) :: k3_sq

    k3_sq = k3_squared(omega, nu, J, lam_c, mu_c)
    k3_val = get_k_from_k_squared(omega, k3_sq)
  end function k3

  function k4(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c) result(k4_val)
    real(rk), intent(in) :: omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    complex(rk) :: k4_val
    complex(rk) :: k4_sq

    k4_sq = k4_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k4_val = get_k_from_k_squared(omega, k4_sq)
  end function k4


end module dispersion_core
