      DOUBLE PRECISION FUNCTION D1MACH(I)
C     Modern implementation using Fortran intrinsics
      IMPLICIT NONE
      INTEGER I

      IF (I .EQ. 1) THEN
C       Smallest positive magnitude
        D1MACH = TINY(1.0D0)
      ELSE IF (I .EQ. 2) THEN
C       Largest magnitude
        D1MACH = HUGE(1.0D0)
      ELSE IF (I .EQ. 3) THEN
C       Smallest relative spacing
        D1MACH = EPSILON(1.0D0) / RADIX(1.0D0)
      ELSE IF (I .EQ. 4) THEN
C       Largest relative spacing
        D1MACH = EPSILON(1.0D0)
      ELSE IF (I .EQ. 5) THEN
C       LOG10(B) where B is the radix
        D1MACH = LOG10(DBLE(RADIX(1.0D0)))
      ELSE
        WRITE(*,*) 'D1MACH: Invalid argument I=', I
        STOP
      END IF

      RETURN
      END
