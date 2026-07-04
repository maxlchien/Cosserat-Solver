      REAL FUNCTION R1MACH(I)
C     Modern implementation using Fortran intrinsics
      IMPLICIT NONE
      INTEGER I

      IF (I .EQ. 1) THEN
C       Smallest positive magnitude
        R1MACH = TINY(1.0)
      ELSE IF (I .EQ. 2) THEN
C       Largest magnitude
        R1MACH = HUGE(1.0)
      ELSE IF (I .EQ. 3) THEN
C       Smallest relative spacing
        R1MACH = EPSILON(1.0) / RADIX(1.0)
      ELSE IF (I .EQ. 4) THEN
C       Largest relative spacing
        R1MACH = EPSILON(1.0)
      ELSE IF (I .EQ. 5) THEN
C       LOG10(B) where B is the radix
        R1MACH = LOG10(REAL(RADIX(1.0)))
      ELSE
        WRITE(*,*) 'R1MACH: Invalid argument I=', I
        STOP
      END IF

      RETURN
      END
