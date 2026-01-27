      INTEGER FUNCTION I1MACH(I)
C     Modern implementation using Fortran intrinsics
      IMPLICIT NONE
      INTEGER I

      IF (I .EQ. 1) THEN
C       Standard input unit
        I1MACH = 5
      ELSE IF (I .EQ. 2) THEN
C       Standard output unit
        I1MACH = 6
      ELSE IF (I .EQ. 3) THEN
C       Standard punch unit (obsolete, use output)
        I1MACH = 6
      ELSE IF (I .EQ. 4) THEN
C       Standard error unit
        I1MACH = 0
      ELSE IF (I .EQ. 5) THEN
C       Bits per integer
        I1MACH = BIT_SIZE(I)
      ELSE IF (I .EQ. 6) THEN
C       Characters per integer (assume 4 bytes = 4 chars)
        I1MACH = 4
      ELSE IF (I .EQ. 7) THEN
C       Base for integers
        I1MACH = RADIX(1)
      ELSE IF (I .EQ. 8) THEN
C       Digits in integer
        I1MACH = DIGITS(1)
      ELSE IF (I .EQ. 9) THEN
C       Largest integer
        I1MACH = HUGE(1)
      ELSE IF (I .EQ. 10) THEN
C       Radix for floating point
        I1MACH = RADIX(1.0D0)
      ELSE IF (I .EQ. 11) THEN
C       Number of base digits in single precision
        I1MACH = DIGITS(1.0)
      ELSE IF (I .EQ. 12) THEN
C       Smallest exponent for single precision
        I1MACH = MINEXPONENT(1.0)
      ELSE IF (I .EQ. 13) THEN
C       Largest exponent for single precision
        I1MACH = MAXEXPONENT(1.0)
      ELSE IF (I .EQ. 14) THEN
C       Number of base digits in double precision
        I1MACH = DIGITS(1.0D0)
      ELSE IF (I .EQ. 15) THEN
C       Smallest exponent for double precision
        I1MACH = MINEXPONENT(1.0D0)
      ELSE IF (I .EQ. 16) THEN
C       Largest exponent for double precision
        I1MACH = MAXEXPONENT(1.0D0)
      ELSE
        WRITE(*,*) 'I1MACH: Invalid argument I=', I
        STOP
      END IF

      RETURN
      END
