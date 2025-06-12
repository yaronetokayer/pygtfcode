class Constants:
    """
    Physical constants used throughout the gravothermal fluid code.
    
    All values are in astrophysically convenient units. Advanced users
    may override these if needed.
    """

    # Gravitational constant in (M_sun^-1 Mpc (km/s)^2)
    gee = 4.2994e-9

    # Conversion of megaparsec to centimeters
    Mpc_to_cm = 3.086e24

    # Conversion of solar mass to grams
    Msun_to_gram = 1.99e33

    # Conversion of seconds to gigayears
    sec_to_Gyr = 3.16881e-17
