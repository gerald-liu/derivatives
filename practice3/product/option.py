import numpy as np
from scipy.stats import norm

from product.constants import *
from product.params import Params


def init_option(product, T, r_min, r_max, X_lo, X_hi, H_lo, H_hi, K, pt, sigma, r, b):
    params = Params(sigma, r, b)
    match product:
        case '简单看涨':
            return EuVanilla(1, T, params, r_min, pt, X_lo, is_call=True)
        case '简单看跌':
            return EuVanilla(1, T, params, r_min, pt, X_hi, is_call=False)
        case '看涨二元':
            return Binary(1, T, params, r_min, r_max, X_hi, is_call=True)
        case '看跌二元':
            return Binary(1, T, params, r_min, r_max, X_hi, is_call=False)
        case '看涨价差':
            return Spread(1, T, params, r_min, pt, X_lo, X_hi, is_call=True)
        case '看跌价差':
            return Spread(1, T, params, r_min, pt, X_lo, X_hi, is_call=False)
        case '欧式看涨鲨鱼鳍':
            return EuSharkFin(1, T, params, r_min, pt, X_hi, H_hi, K, is_call=True)
        case '欧式看跌鲨鱼鳍':
            return EuSharkFin(1, T, params, r_min, pt, X_lo, H_lo, K, is_call=False)
        case '欧式双向鲨鱼鳍':
            return DbEuSharkFin(1, T, params, r_min, pt, X_lo, X_hi, H_lo, H_hi, K)
        case '蝶式':
            return Butterfly(1, T, params, r_min, pt, X_lo, X_hi, H_lo, H_hi)
        case '看涨美式触碰':
            return BarCashBinary(1, T, params, r_min, X_hi, K, True, True)
        case '看跌美式触碰':
            return BarCashBinary(1, T, params, r_min, X_lo, K, False, True)
        case '向下敲入看涨':
            return BarVanilla(1, T, params, r_min, pt, X_hi, H_lo, False, True, True)
        case '向上敲入看跌':
            return BarVanilla(1, T, params, r_min, pt, X_lo, H_hi, True, True, False)
        case '看涨鲨鱼鳍':
            return AmSharkFin(1, T, params, r_min, pt, X_hi, H_hi, K, True)
        case '看跌鲨鱼鳍':
            return AmSharkFin(1, T, params, r_min, pt, X_lo, H_lo, K, False)
        case '双向鲨鱼鳍':
            return DbAmSharkFin(1, T, params, r_min, pt, X_lo, X_hi, H_lo, H_hi, K)

class Option():
    def __init__(self, S, T, params: Params, r_min):
        # spot price
        self.S = S
        # time to maturity
        self.T = T
        # (volatility, rate, dividend) = (sigma, r, r - coc)
        self.params = params
        self.r_min = r_min # min return

    @property
    def sigma(self): return self.params.sigma
    @property
    def r(self): return self.params.r
    @property
    def b(self): return self.params.coc
    @property
    def q(self): return self.params.q

    @property
    def vol_T(self): return self.sigma * np.sqrt(self.T)

    def copy_attr(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S = self.S + dS if S is None else S + dS
        T = self.T + dT if T is None else T + dT
        params = Params(self.sigma + dsigma, self.r, self.b)
        return (S, T, params)
    
    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0): # virtual
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return Option(S, T, params, self.r_min)

    def price_r_min(self): return self.r_min * np.exp(- self.r * self.T)
    
    def raw_price(self): raise NotImplementedError()

    def price(self): raise NotImplementedError()


class EuVanilla(Option):
    def __init__(
        self, S, T, params: Params, r_min, pt, X, is_call
    ):
        super().__init__(S, T, params, r_min)
        self.pt = pt # participation rate
        self.X = X # strike
        self.is_call = is_call
        self.phi = 1 if is_call else -1

    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return EuVanilla(S, T, params, self.r_min, self.pt, self.X, self.is_call)

    def raw_price(self):
        S, T, X = self.S, self.T, self.X
        sigma, r, b, q = self.params.unpack()
        phi, vol = self.phi, self.vol_T

        mu = b / sigma**2 - 0.5 # mu = (r-q - sigma^2/2) / sigma^2
        d1 = np.log(S/X) / vol + (1+mu) * vol
        d2 = d1 - vol

        delta = phi * np.exp(-q*T) * norm.cdf(phi * d1)
        proba_ITM = norm.cdf(phi * d2)

        # S*np.exp(-q*T)*norm.cdf(phi*d1) - X*np.exp(-r*T)*norm.cdf(phi*(d1-vol))
        return S * delta - phi * X * np.exp(-r*T) * proba_ITM
    
    def price(self):
        return self.raw_price() * self.pt + self.price_r_min()


class Binary(Option):
    def __init__(self, S, T, params: Params, r_min, r_max, X, is_call):
        super().__init__(S, T, params, r_min)
        self.r_max = r_max
        self.X = X # strike
        self.is_call = is_call
        self.phi = 1 if is_call else -1

    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return Binary(S, T, params, self.r_min, self.r_max, self.X, self.is_call)
    
    def raw_price(self):
        S, T, X = self.S, self.T, self.X
        sigma, r, b, _ = self.params.unpack()
        phi, vol = self.phi, self.vol_T

        mu = b / sigma**2 - 0.5 # mu = (r-q - sigma^2/2) / sigma^2
        d2 = np.log(S/X) / vol + mu * vol

        proba_ITM = norm.cdf(phi * d2)

        # np.exp(-r*T)*norm.cdf(phi*d2)
        return np.exp(-r*T) * proba_ITM
    
    def price(self):
        return self.raw_price() * (self.r_max - self.r_min) + self.price_r_min()


class Spread(Option):
    def __init__(
        self, S, T, params: Params, r_min, pt, X_lo, X_hi, is_call
    ):
        super().__init__(S, T, params, r_min)
        self.pt = pt # participation rate
        self.X_lo = X_lo # low strike
        self.X_hi = X_hi # high strike
        self.is_call = is_call
        self.phi = 1 if is_call else -1

        self.vanilla_lo = EuVanilla(S, T, params, 0, 1, X_lo, is_call)
        self.vanilla_hi = EuVanilla(S, T, params, 0, 1, X_hi, is_call)
    
    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return Spread(S, T, params, self.r_min, self.pt, self.X_lo, self.X_hi, self.is_call)
    
    def raw_price(self):
        return self.phi * (self.vanilla_lo.raw_price() - self.vanilla_hi.raw_price())

    def price(self):
        return self.raw_price() * self.pt + self.price_r_min()


class Straddle(Option):
    def __init__(self, S, T, params: Params, r_min, pt_l, pt_r, X):
        super().__init__(S, T, params, r_min)
        self.pt_l = pt_l
        self.pt_r = pt_r
        self.X = X

        self.vanilla_put = EuVanilla(S, T, params, 0, 1, X, False)
        self.vanilla_call = EuVanilla(S, T, params, 0, 1, X, True)
    
    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return Straddle(S, T, params, self.r_min, self.pt_l, self.pt_r, self.X)

    def raw_price(self):
        return self.vanilla_put.raw_price() + self.vanilla_call.raw_price()
    
    def price(self):
        return (
            self.vanilla_put.raw_price() * self.pt_l + 
            self.vanilla_call.raw_price() * self.pt_r + 
            self.price_r_min()
        )

class EuSharkFin(Option):
    def __init__(self, S, T, params: Params, r_min, pt, X, H, K, is_call):
        super().__init__(S, T, params, r_min)
        self.pt = pt # participation rate
        self.X = X # strike
        self.H = H # barrier
        self.K = K # rebate
        self.is_call = is_call
        self.phi = 1 if is_call else -1

        self.Spread = Option(S, T, params, 0) # virtual
        if is_call:
            self.Spread = Spread(S, T, params, 0, pt, X, H, is_call)
        else:
            self.Spread = Spread(S, T, params, 0, pt, H, X, is_call)
        
        r_max_bin = self.phi * (H - X) + r_min - K
        self.Binary = Binary(S, T, params, 0, r_max_bin, H, is_call)
    
    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return EuSharkFin(S, T, params, self.r_min, self.pt, self.X, self.H, self.K, self.is_call)
    
    def price(self):
        return (
            self.Spread.price() 
            - self.Binary.price() + 
            self.price_r_min()
        )


class DbEuSharkFin(Option):
    def __init__(self, S, T, params: Params, r_min, pt, X_lo, X_hi, H_lo, H_hi, K):
        super().__init__(S, T, params, r_min)
        self.pt = pt # participation rate
        self.X_lo = X_lo # put strike
        self.H_lo = H_lo # put barrier
        self.X_hi = X_hi # call strike
        self.H_hi = H_hi # call barrier
        self.K = K # rebate

        self.EuSharkFin_put = EuSharkFin(S, T, params, 0, pt, X_lo, H_lo, K, False) # LHS
        self.EuSharkFin_call = EuSharkFin(S, T, params, 0, pt, X_hi, H_hi, K, True) # RHS

    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return DbEuSharkFin(S, T, params, self.r_min, self.pt, self.X_lo, self.X_hi, self.H_lo, self.H_hi, self.K)

    def price(self):
        return (
            self.EuSharkFin_put.price() + 
            self.EuSharkFin_call.price() + 
            self.price_r_min()
        )
    

class Butterfly(Option):
    def __init__(self, S, T, params: Params, r_min, pt, X_lo, X_hi, H_lo, H_hi):
        super().__init__(S, T, params, r_min)
        self.pt = pt # participation rate
        self.X_lo = X_lo # call barrier
        self.H_lo = H_lo # call strike
        self.X_hi = X_hi # put barrier
        self.H_hi = H_hi # put strike

        self.EuSharkFin_call = EuSharkFin(S, T, params, 0, pt, H_lo, X_lo, 0, True) # LHS
        self.EuSharkFin_put = EuSharkFin(S, T, params, 0, pt, H_hi, X_hi, 0, False) # RHS
    
    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return Butterfly(S, T, params, self.r_min, self.pt, self.X_lo, self.X_hi, self.H_lo, self.H_hi)

    def price(self):
        return (
            self.EuSharkFin_call.price() + 
            self.EuSharkFin_put.price() + 
            self.price_r_min()
        )


# cash at expiration
# American one-touch: up-in or down-in
class BarCashBinary(Option):
    def __init__(
        self, S, T, params: Params, r_min, H, K, up, ki, dt=DT
    ):
        super().__init__(S, T, params, r_min)
        self.H = H # barrier
        self.K = K # cash at expiration
        self.up = up
        self.ki = ki
        self.dt = dt
    
    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return BarCashBinary(S, T, params, self.r_min, self.H, self.K, self.up, self.ki)

    def raw_price(self):     
        S, T = self.S, self.T
        K = self.K - self.r_min
        sigma, r, b, _ = self.params.unpack()

        # knocked-in or knocked out
        if self.ki:
            if (self.up and S >= self.H) or ((not self.up) and S <= self.H):
                return K * np.exp(-r*T)
        else:
            if (self.up and S >= self.H) or ((not self.up) and S <= self.H):
                return 0

        sgn = 1 if self.up else -1
        H = self.H * np.exp(sgn * BETA * self.sigma * np.sqrt(self.dt))
        
        vol = self.vol_T
        mu = b / sigma**2 - 0.5 # mu = (r-q - sigma^2/2) / sigma^2

        if self.up:
            eta = -1
            phi = 1 if self.ki else -1
        else:
            eta = 1
            phi = -1 if self.ki else 1

        x2 = np.log(S/H) / vol + (1+mu) * vol # similar to d1, but X changed to H
        y2 = np.log(H/S) / vol + (1+mu) * vol
        
        B2 = K * np.exp(-r*T) * norm.cdf(phi*(x2-vol))
        B4 = K * np.exp(-r*T) * (H/S)**(2*mu) * norm.cdf(eta*(y2-vol))

        if self.ki:
            return B2 + B4
        else:
            return B2 - B4

    def price(self):
        return self.raw_price() + self.price_r_min()


class BarVanilla(Option):
    def __init__(
        self, S, T, params: Params, r_min, pt, X, H, up, ki, is_call, dt=DT
    ):
        super().__init__(S, T, params, r_min)
        self.pt = pt
        self.X = X # strike
        self.H = H # barrier
        self.up = up
        self.ki = ki
        self.is_call = is_call
        self.dt = dt
    
    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return BarVanilla(S, T, params, self.r_min, self.pt, self.X, self.H, self.up, self.ki, self.is_call)

    def raw_price(self):
        S, T, X = self.S, self.T, self.X

        # knocked-in or knocked out
        if self.ki:
            if (self.up and S >= self.H) or ((not self.up) and S <= self.H):
                vanilla = EuVanilla(S, T, self.params, 0, 1, X, self.is_call)
                return vanilla.raw_price()
        else:
            if (self.up and S >= self.H) or ((not self.up) and S <= self.H):
                return 0

        sigma, r, b, q = self.params.unpack()

        sgn = 1 if self.up else -1
        H = self.H * np.exp(sgn * BETA * self.sigma * np.sqrt(self.dt))

        vol = self.vol_T     
        mu = b / sigma**2 - 0.5 # mu = (coc - sigma^2/2) / sigma^2

        x1 = np.log(S/X) / vol + (1+mu) * vol # d1
        x2 = np.log(S/H) / vol + (1+mu) * vol

        y1 = np.log(H/S * H/X) / vol + (1+mu) * vol
        y2 = np.log(H/S) / vol + (1+mu) * vol

        eta = -1 if self.up else 1
        phi = 1 if self.is_call else -1
        
        A = S * np.exp(-q*T) * norm.cdf(phi*x1) \
            - X * np.exp(-r*T) * norm.cdf(phi*(x1-vol))
        
        B = S * np.exp(-q*T) * norm.cdf(phi*x2) \
            - X * np.exp(-r*T) * norm.cdf(phi*(x2-vol))
        
        C = S * np.exp(-q*T) * (H/S)**(2*mu+2) * norm.cdf(eta*y1) \
            - X * np.exp(-r*T) * (H/S)**(2*mu) * norm.cdf(eta*(y1-vol))
        
        D = S * np.exp(-q*T) * (H/S)**(2*mu+2) * norm.cdf(eta*y2) \
            - X * np.exp(-r*T) * (H/S)**(2*mu) * norm.cdf(eta*(y2-vol))

        A *= phi
        B *= phi
        C *= phi
        D *= phi

        if self.ki:
            if (not self.up) and self.is_call: # down-in call
                return C if X > H else A - B + D
            if self.up and (not self.is_call): # up-in put
                return C if X < H else A - B + D
            if self.up and self.is_call: # up-in call
                return B - C + D if X < H else A
            if (not self.up) and (not self.is_call): # down-in put
                return B - C + D if X > H else A
        else:
            if (not self.up) and self.is_call: # down-out call
                return A - C if X > H else B - D
            if self.up and (not self.is_call): # up-out put
                return A - C if X < H else B - D
            if self.up and self.is_call: # up-out call
                return A - B + C - D if X < H else 0
            if (not self.up) and (not self.is_call): # down-out put
                return A - B + C - D if X > H else 0
    
    def price(self):
        return self.raw_price() * self.pt + self.price_r_min()


class AmSharkFin(Option):
    def __init__(
        self, S, T, params: Params, r_min, pt, X, H, K, is_call, dt=DT
    ):
        super().__init__(S, T, params, r_min)
        self.pt = pt
        self.X = X # strike
        self.H = H # barrier
        self.K = K # rebate
        self.is_call = is_call
        self.dt = dt

        self.BarVanilla = Option(S, T, params, 0) # virtual
        self.BarCashBinary = Option(S, T, params, 0) # virtual
        if is_call:
            self.BarVanilla = BarVanilla(S, T, params, 0, 1, X, H, True, False, True)
            self.BarCashBinary = BarCashBinary(S, T, params, 0, H, K-r_min, True, True)
        else:
            self.BarVanilla = BarVanilla(S, T, params, 0, 1, X, H, False, False, False)
            self.BarCashBinary = BarCashBinary(S, T, params, 0, H, K-r_min, False, True)

    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return AmSharkFin(S, T, params, self.r_min, self.pt, self.X, self.H, self.K, self.is_call)

    def price(self):
        return (
            self.BarVanilla.raw_price() * self.pt + 
            self.BarCashBinary.raw_price() + 
            self.price_r_min()
        )
    

class DbBarCashBinary(Option):
    def __init__(self, S, T, params: Params, r_min, L, U, K, ki, dt=DT):
        super().__init__(S, T, params, r_min)
        self.L = L # lower barrier
        self.U = U # upper barrier
        self.K = K # rebate
        self.ki = ki
        self.dt = dt

    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return DbBarCashBinary(S, T, params, self.r_min, self.L, self.U, self.K, self.ki)

    def raw_price(self, N=5):
        S, T, params = self.S, self.T, self.params
        K = self.K - self.r_min
        sigma, r, b, _ = params.unpack()

        # knocked in or knocked out
        if self.ki and (S <= self.L or S >= self.U):
            return K * np.exp(-r*T)
        elif (not self.ki) and (S <= self.L or S >= self.U):
            return 0

        L = self.L * np.exp(- BETA * self.sigma * np.sqrt(self.dt))
        U = self.U * np.exp(BETA * self.sigma * np.sqrt(self.dt))

        vol = self.vol_T
        Z = np.log(U/L)
        alpha = 0.5 - b / sigma**2 # -0.5*(2b/sigma^2 - 1) = -mu 
        beta = - alpha**2 - 2*r / sigma**2

        sum = 0

        for i in range(1, N + 1):
            ipz = i * np.pi / Z

            sum += i * ((S/L)**alpha - (-1)**i * (S/U)**alpha) / (alpha**2 + ipz**2) * \
                   np.sin(ipz * np.log(S/L)) * np.exp(- (ipz**2 - beta) * vol**2 / 2)

        price_db_out = 2 * np.pi * K / Z**2 * sum

        if not self.ki:
            return price_db_out
        else:
            return K * np.exp(-r*T) - price_db_out

    def price(self):
        return self.raw_price() + self.r_min


# strike X is inside the barrier range (L, U)
class DbBarVanilla(Option):
    def __init__(self, S, T, params: Params, r_min, pt, X, L, U, ki, is_call, dt=DT):
        super().__init__(S, T, params, r_min)
        self.pt = pt
        self.X = X # strike
        self.L = L # lower barrier
        self.U = U # upper barrier
        self.ki = ki
        self.is_call = is_call
        self.dt = dt

        if X < L or X > U:
            raise Exception('Strike out of barrier range. Use a different class.')

    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return DbBarVanilla(S, T, params, self.r_min, self.pt, self.X, self.L, self.U, self.ki, self.is_call)

    def raw_price(self, N=5):
        S, T, X, params = self.S, self.T, self.X, self.params
        sigma, r, b, q = params.unpack()

        vanilla = EuVanilla(S, T, params, 0, 1, X, self.is_call)

        # knocked in or knocked out
        if self.ki and (S <= self.L or S >= self.U):
            return vanilla.raw_price()
        elif (not self.ki) and (S <= self.L or S >= self.U):
            return 0

        L = self.L * np.exp(- BETA * self.sigma * np.sqrt(self.dt))
        U = self.U * np.exp(BETA * self.sigma * np.sqrt(self.dt))

        """
        flat boundaries: delta1 = delta2 = 0
        mu2 = 0, (L/S)**mu2 = 1 is ignored
        F = U * e^(delta1 * T) = U (call)
        E = L * e^(delta2 * T) = L (put)
        """
        vol = self.vol_T
        mu = b / sigma**2 - 0.5 # mu = (coc - sigma^2/2) / sigma^2
        mu1 = 2 * b / sigma**2 + 1
        mu3 = 2 * b / sigma**2 + 1
        
        F = U if self.is_call else L
        phi = 1 if self.is_call else -1

        sum_S = 0
        sum_X = 0

        for n in range(- N, N + 1):
            ULn = (U/L)**n

            d1 = np.log(S/X * ULn**2) / vol + (1+mu) * vol
            d2 = np.log(S/F * ULn**2) / vol + (1+mu) * vol
            d3 = np.log(L/S * L/X / ULn**2) / vol + (1+mu) * vol     
            d4 = np.log(L/S * L/F / ULn**2) / vol + (1+mu) * vol

            sum_S += ULn**mu1 * phi * (norm.cdf(d1) - norm.cdf(d2)) \
                     - (L/S / ULn)**mu3 * phi * (norm.cdf(d3) - norm.cdf(d4))
            
            sum_X += ULn**(mu1-2) * phi * (norm.cdf(d1-vol) - norm.cdf(d2-vol)) \
                     - (L/S / ULn)**(mu3-2) * phi * (norm.cdf(d3-vol) - norm.cdf(d4-vol))
        
        price_db_out = phi * (S * np.exp(-q*T) * sum_S - X * np.exp(-r*T) * sum_X)

        if not self.ki:
            return price_db_out
        else:
            return vanilla.raw_price() - price_db_out
    
    def price(self):
        return self.raw_price() * self.pt + self.price_r_min()
    

class DbAmSharkFin(Option):
    def __init__(self, S, T, params: Params, r_min, pt, X_lo, X_hi, H_lo, H_hi, K, dt=DT):
        super().__init__(S, T, params, r_min)
        self.pt = pt
        self.X_lo = X_lo # lower strike
        self.X_hi = X_hi # higher strike
        self.H_lo = H_lo # lower barrier
        self.H_hi = H_hi # upper barrier
        self.K = K # rebate
        self.dt = dt

        self.DbOutPut = DbBarVanilla(S, T, params, 0, 1, X_lo, H_lo, H_hi, False, False)
        self.DbOutCall = DbBarVanilla(S, T, params, 0, 1, X_hi, H_lo, H_hi, False, True) 
        self.DbInBinary = DbBarCashBinary(S, T, params, 0, H_lo, H_hi, K-r_min, True)

    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        S, T, params = self.copy_attr(S, T, dS, dT, dsigma)
        return DbAmSharkFin(S, T, params, self.r_min, self.pt, self.X_lo, self.X_hi, self.H_lo, self.H_hi, self.K)

    def price(self):
        return (
            self.DbOutPut.raw_price() * self.pt + 
            self.DbOutCall.raw_price() * self.pt + 
            self.DbInBinary.raw_price() + 
            self.price_r_min()
        )
