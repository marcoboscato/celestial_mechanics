      SUBROUTINE JJDATE (DJJ,JR,MS,LAN,LH,MN,SEC)
*     =============================================================
*     Julian date => Gregorian date

*     (c) copyright Bureau des longitudes 1995
*     =============================================================


*     CONVERSION D'UNE DATE  DE LA PERIODE JULIENNE EN DATE  GREGORIENNE
*     DATES POSTERIEURS AU 1 JANVIER 1601   ( 2305813.5  JOURS JULIENS )
*     DJ EST INITIALISE SELON LA VALEUR DE DJJ

      IMPLICIT REAL*8 (D-D,R-S)

      INI=1
      CALL INITI(DJJ,DJ,IAN,LAN,INI)
      DDI=0.D0
      IF(DJJ-DJ) 8,8,9
  9   IAN=IAN+1
      DJ=DJ+FLOAT(NJA(IAN-1))
      IF(DJJ-DJ) 7,8,9
   7  LAN=IAN-1
      DI=DJJ-(DJ-FLOAT(NJA(LAN)))
      DDI=DI-FLOAT(IDINT(DI))
      IDI=IDINT(DI)
      IM=0
      NJE=0
  12  IM=IM+1
      NJE=NJE+NJM(IM,LAN)
      IF(IDI-NJE) 10,11,12
  10  MS=IM
      IF(MS-1) 18,18,19
  18  JR=IDI+1
      GO TO 20
  19  JR=IDI- NJE+NJM(IM,LAN)+1
  20  LH=IDINT(DDI*24.D0+0.000001D0)
      RH=DDI*24.D0-FLOAT(LH)
      MN=IDINT(RH*60.D0+0.00001D0)
      RM=RH*60.D0-FLOAT(MN)
      SEC=RM*60.D0
      SEC=DABS(SEC)
      GO TO 3
   8  LAN=IAN
      MS=1
      GO TO 13
  11  MS=IM+1
  13  JR=1
      IF(DDI.NE.0.D0) GO TO 20
      LH=0
      MN=0
      SEC=0.
   3  RETURN
      END

      SUBROUTINE INITI(DJJ,DJ,IAN,LAN,I)
*     =============================================================
*     Initial date for the computation of Gregorian dates or Julian dates

*     (c) copyright Bureau des longitudes 1995
*     =============================================================


*     DETERMINATION DE LA DATE DE DEPART POUR LE CALCUL DE LA DATE
*     GREGORIENNE ( I=1 )  OU DE LA DATE JULIENNE ( I=2 )

      IMPLICIT REAL *8 (D-D)
      DIMENSION INI(16)

      DATA INI/5813,24075,42337,60599,78861,97123,115385,133647,151910,
     &  170172,188434,206696,224958,243220,261482,279744/
      GO TO (10,20),I
  10  IDI=IDINT(DJJ-2300000.5D0)
      DO 1 I=1,16
      IF(IDI.LT.INI(I)) GO TO 1
      DJ=2300000.5D0+FLOAT(INI(I))
      IAN=1601+(I-1)*50
   1  CONTINUE
      RETURN
  20  CONTINUE
      DO 2 I=1,16
      JAN=1601+(I-1)*50
      IF(LAN.LT.JAN) GO TO 2
      DJ=2300000.5D0+FLOAT(INI(I))
      IAN=JAN
   2  CONTINUE
      RETURN
      END

      FUNCTION NJM(IM,LAN)
      DIMENSION NJOURS(12)
      DATA NJOURS/31,28,31,30,31,30,31,31,30,31,30,31/
      NJM=NJOURS(IM)
      IF(IM.NE.2) RETURN
      A=FLOAT(LAN)
      IE=INT(A/4.+0.0001)
      IE=IE*4
      IF((LAN-IE).EQ.0) NJM=29
      IE=INT(A/100.+0.00001)
      IE=IE*100
      IF((LAN-IE).NE.0) RETURN
      IE=IE/100
      E=FLOAT(IE)
      JE=INT(E/4.+0.0001)
      JE=JE*4
      IF((IE-JE).EQ.0) RETURN
      NJM=28
      RETURN
      END

      FUNCTION NJA(LAN)
      NJA=337+NJM(2,LAN)
      RETURN
      END


      SUBROUTINE DATEJJ(JR,MS,LAN,LH,MN,SEC,DJJ)
*     =============================================================
*     Gregorian date => Julian date

*     (c) copyright Bureau des longitudes 1995
*     =============================================================

*     CONVERSION D'UNE DATE GREGORIENNE EN DATE EN JOURS DE LA PERIODE JULIENNE
*     DATES POSTERIEURS AU 1 JANVIER 1601

*     DJ EST INITIALISE SELON LA VALEUR DE LAN

      DOUBLE PRECISION DJJ,SEC,DJ

      INI=2
      CALL INITI(DJJ,DJ,IAN,LAN,INI)
      NAN=LAN-1
      DJJ=DJ
      IF(IAN.EQ.LAN) GO TO 2
      DO 6 I=IAN,NAN
      NJ=NJA(I)
      DJJ=DJJ+FLOAT(NJ)
   6  CONTINUE
   2  CONTINUE
      IF(MS.EQ.1) GO TO 3
      MSM=MS-1
      DO 16 IM=1,MSM
      NJ=NJM(IM,LAN)
      DJJ=DJJ+FLOAT(NJ)
  16  CONTINUE
  3   DJJ=DJJ+FLOAT(JR-1)
      DJJ=DJJ+FLOAT(LH)/24.D0
      DJJ=DJJ+FLOAT(MN)/1440.D0
      DJJ=DJJ+SEC/86400.D0
      RETURN
      END
