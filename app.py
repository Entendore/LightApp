#!/usr/bin/env python3
"""
✨ Light & Color Learning App — GPU-Accelerated, Feature-Rich Edition
Interactive educational application exploring the physics of light and color.
Uses Numba JIT + CuPy GPU acceleration for real-time optical simulations.
Built with PySide6.

Install:
    pip install PySide6 numba cupy-cuda12x   # GPU path
    pip install PySide6 numba numpy           # CPU fallback
"""

import sys, math, random, time
import numpy as np
from numba import njit, prange

# ── GPU Backend ────────────────────────────────────────────────────────
try:
    import cupy as cp
    HAS_CUPY = True
    xp = cp
except ImportError:
    cp = None
    HAS_CUPY = False
    xp = np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QPushButton, QGroupBox, QListWidget,
    QStackedWidget, QGridLayout, QRadioButton, QButtonGroup, QFrame,
    QDoubleSpinBox, QSpinBox, QTabWidget, QLineEdit, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QRadialGradient,
    QLinearGradient, QPainterPath, QPolygonF, QImage, QPixmap
)

# ═══════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
C      = 299_792_458
H      = 6.626e-34
HBAR   = 1.0546e-34
KB     = 1.381e-23
EV     = 1.602e-19
WIEN_B = 2.898e-3       # Wien displacement constant (m·K)
STEFAN = 5.670e-8       # Stefan-Boltzmann constant (W/m²/K⁴)
MEDIA  = {
    "Vacuum": 1.0000, "Air": 1.0003, "Water": 1.3330,
    "Glass (Crown)": 1.5200, "Glass (Flint)": 1.6200,
    "Diamond": 2.4170, "Ice": 1.3100, "Acrylic": 1.4900,
    "Quartz": 1.5440, "Sapphire": 1.7700, "Dense Flint": 1.7500,
}

# ═══════════════════════════════════════════════════════════════════════
# NUMBA-ACCELERATED FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def wavelength_to_rgb_numba(wl):
    gamma = 0.8
    wl = max(380.0, min(780.0, wl))
    if wl < 440:     r, g, b = -(wl-440)/60., 0., 1.
    elif wl < 490:   r, g, b = 0., (wl-440)/50., 1.
    elif wl < 510:   r, g, b = 0., 1., -(wl-510)/20.
    elif wl < 580:   r, g, b = (wl-510)/70., 1., 0.
    elif wl < 645:   r, g, b = 1., -(wl-645)/65., 0.
    else:            r, g, b = 1., 0., 0.
    if wl < 420:     f = 0.3 + 0.7*(wl-380)/40.
    elif wl <= 700:  f = 1.
    elif wl <= 780:  f = 0.3 + 0.7*(780-wl)/80.
    else:            f = 0.
    return (int(255*(r*f)**gamma), int(255*(g*f)**gamma), int(255*(b*f)**gamma))


@njit(parallel=True, cache=True)
def generate_spectrum_line(width, wl_min=380., wl_max=780.):
    out = np.empty((width, 3), dtype=np.uint8)
    for i in prange(width):
        wl = wl_min + (wl_max-wl_min)*i/(width-1)
        r, g, b = wavelength_to_rgb_numba(wl)
        out[i,0]=r; out[i,1]=g; out[i,2]=b
    return out


@njit(parallel=True, cache=True)
def generate_diffraction_1d_numba(y_coords, lam, a, d, N_slits):
    n = y_coords.shape[0]; intensity = np.empty(n, dtype=np.float64)
    for i in prange(n):
        y = y_coords[i]
        beta = math.pi*a*y/(lam*1.)
        sb = math.sin(beta)/beta if abs(beta) > 1e-12 else 1.
        diff = sb*sb
        if N_slits <= 1: interf = 1.
        elif N_slits == 2:
            delta = math.pi*d*y/(lam*1.); interf = math.cos(delta)**2
        else:
            delta = math.pi*d*y/(lam*1.); sd = math.sin(delta)
            interf = 1. if abs(sd)<1e-12 else (math.sin(N_slits*delta)/(N_slits*sd))**2
        intensity[i] = diff*interf
    return intensity


@njit(parallel=True, cache=True)
def planck_spectrum_numba(wl_m, T):
    """Planck's law: spectral radiance for each wavelength."""
    n = wl_m.shape[0]; B = np.empty(n, dtype=np.float64)
    c1 = 2.*H*C*C; c2 = H*C/KB
    for i in prange(n):
        lam = wl_m[i]
        if lam <= 0: B[i] = 0.; continue
        ex = c2/(lam*T)
        if ex > 500: B[i] = 0.
        else: B[i] = c1/(lam**5 * (math.exp(ex)-1.))
    return B


@njit(parallel=True, cache=True)
def fresnel_coefficients_numba(angles_deg, n1, n2):
    """Compute Rs, Rp, Ts, Tp for array of angles."""
    n = angles_deg.shape[0]
    Rs = np.empty(n, dtype=np.float64)
    Rp = np.empty(n, dtype=np.float64)
    Ts = np.empty(n, dtype=np.float64)
    Tp = np.empty(n, dtype=np.float64)
    for i in prange(n):
        th = math.radians(angles_deg[i])
        sin2 = (n1/n2*math.sin(th))**2
        if sin2 > 1.:          # TIR
            Rs[i]=1.; Rp[i]=1.; Ts[i]=0.; Tp[i]=0.
        else:
            cos_i = math.cos(th)
            cos_t = math.sqrt(1.-sin2)
            rs = (n1*cos_i - n2*cos_t)/(n1*cos_i + n2*cos_t)
            rp = (n2*cos_i - n1*cos_t)/(n2*cos_i + n1*cos_t)
            Rs[i] = rs*rs; Rp[i] = rp*rp
            Ts[i] = 1.-Rs[i]; Tp[i] = 1.-Rp[i]
    return Rs, Rp, Ts, Tp


@njit(parallel=True, cache=True)
def rayleigh_scattering_numba(wl_nm_array):
    """Relative Rayleigh scattering intensity ∝ 1/λ⁴."""
    n = wl_nm_array.shape[0]; scat = np.empty(n, dtype=np.float64)
    ref = 550.  # normalize to green
    for i in prange(n):
        scat[i] = (ref / wl_nm_array[i])**4
    return scat


# ═══════════════════════════════════════════════════════════════════════
# GPU (CuPy) COMPUTE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def gpu_compute_diffraction_2d(pattern, wl_nm, slit_w_um, slit_d_um, n_slits, res=400):
    lam=wl_nm*1e-9; a=slit_w_um*1e-6; d=slit_d_um*1e-6; half=0.03
    y=xp.linspace(-half,half,res); X,Y=xp.meshgrid(y,y)
    beta=xp.pi*a*Y/lam; sb=xp.where(xp.abs(beta)<1e-12,1.,xp.sin(beta)/beta); diff=sb**2
    if pattern==0: interf=xp.ones_like(Y)
    elif pattern==1: delta=xp.pi*d*Y/lam; interf=xp.cos(delta)**2
    else: delta=xp.pi*d*Y/lam; sd=xp.sin(delta); interf=xp.where(xp.abs(sd)<1e-12,1.,(xp.sin(n_slits*delta)/(n_slits*sd))**2)
    h=a*40; bx=xp.pi*h*X/lam; sbx=xp.where(xp.abs(bx)<1e-12,1.,xp.sin(bx)/bx)
    I=diff*interf*sbx**2
    return I.get() if HAS_CUPY and isinstance(I,cp.ndarray) else np.asarray(I)

def gpu_compute_wave_field_2d(sources, phase, res=300):
    half=10.; x=xp.linspace(-half,half,res); y=xp.linspace(-half,half,res)
    X,Y=xp.meshgrid(x,y); psi=xp.zeros((res,res),dtype=xp.float64)
    for sx,sy in sources:
        r=xp.sqrt((X-sx)**2+(Y-sy)**2); r=xp.maximum(r,0.05)
        psi+=xp.sin(2*xp.pi*r-phase)/xp.sqrt(r)
    mx=xp.max(xp.abs(psi))
    if mx>0: psi/=mx
    return psi.get() if HAS_CUPY and isinstance(psi,cp.ndarray) else np.asarray(psi)

def gpu_compute_blackbody_2d(T, res=300):
    """2D blackbody color bar across wavelengths on GPU."""
    wl = xp.linspace(100e-9, 3000e-9, res)  # 100nm to 3μm
    c1 = 2*H*C**2; c2 = H*C/KB
    ex = c2/(wl*T)
    B = c1/(wl**5 * (xp.exp(ex)-1.))
    # Tile into 2D image
    img = xp.tile(B, (res//4, 1))
    return img.get() if HAS_CUPY and isinstance(img,cp.ndarray) else np.asarray(img)


# ═══════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def wavelength_to_rgb(wl):
    r,g,b = wavelength_to_rgb_numba(float(wl)); return QColor(r,g,b)

def color_name(wl):
    if   wl<380: return "Ultraviolet"
    elif wl<450: return "Violet"
    elif wl<495: return "Blue"
    elif wl<570: return "Green"
    elif wl<590: return "Yellow"
    elif wl<620: return "Orange"
    elif wl<780: return "Red"
    else:        return "Infrared"

def wl_to_freq(nm):   return C/(nm*1e-9)
def wl_to_energy(nm): return H*C/(nm*1e-9)/EV

def em_region(wl_nm):
    if wl_nm < 0.01:   return "Gamma rays"
    elif wl_nm < 10:   return "X-rays"
    elif wl_nm < 380:  return "Ultraviolet"
    elif wl_nm < 780:  return "Visible light"
    elif wl_nm < 1e6:  return "Infrared"
    elif wl_nm < 1e9:  return "Microwaves"
    else:              return "Radio waves"

def blackbody_rgb(T):
    """Attempt approximate RGB color of blackbody at temperature T."""
    # Use CIE 1931 approximate method
    t = T / 100.
    # Red
    if t <= 66: r = 255
    else: r = 329.698727446 * ((t-60)**-0.1332047592)
    # Green
    if t <= 66: g = 99.4708025861 * math.log(t) - 161.1195681661
    else: g = 288.1221695283 * ((t-60)**-0.0755148492)
    # Blue
    if t >= 66: b = 255
    elif t <= 19: b = 0
    else: b = 138.5177312231 * math.log(t-10) - 305.0447927307
    return QColor(int(max(0,min(255,r))), int(max(0,min(255,g))), int(max(0,min(255,b))))

def heading(txt, sz=18):
    l=QLabel(txt); l.setStyleSheet(f"font-size:{sz}px;font-weight:bold;color:#e0e0f0;"); return l

def body(txt):
    l=QLabel(txt); l.setWordWrap(True)
    l.setStyleSheet("font-size:13px;color:#b0b0c8;line-height:1.5;"); return l

def info_label(txt=""):
    l=QLabel(txt); l.setWordWrap(True)
    l.setStyleSheet("background:#12122a;border:1px solid #2a2a5a;border-radius:6px;"
                    "padding:10px;font-size:13px;color:#c8c8e0;"); return l

def make_slider(lo, hi, val, cb):
    s=QSlider(Qt.Horizontal); s.setRange(lo,hi); s.setValue(val); s.valueChanged.connect(cb); return s

def intensity_to_rgba(I, base_rgb, gamma=0.35):
    h,w = I.shape; bright = np.power(np.clip(I,0,1),gamma)
    rgba = np.empty((h,w,4), dtype=np.uint8)
    rgba[:,:,0]=(bright*base_rgb[0]).astype(np.uint8)
    rgba[:,:,1]=(bright*base_rgb[1]).astype(np.uint8)
    rgba[:,:,2]=(bright*base_rgb[2]).astype(np.uint8)
    rgba[:,:,3]=255; return rgba

def wave_to_rgba(field):
    h,w=field.shape; mx=np.max(np.abs(field)); n=field/mx if mx>0 else field
    pos=np.clip(n,0,1); neg=np.clip(-n,0,1)
    rgba=np.empty((h,w,4),dtype=np.uint8)
    rgba[:,:,0]=(pos*255+neg*15).astype(np.uint8)
    rgba[:,:,1]=(pos*120+neg*60).astype(np.uint8)
    rgba[:,:,2]=(pos*20+neg*230).astype(np.uint8)
    rgba[:,:,3]=255; return rgba

def spectrum_curve_to_rgba(wl_nm, B, width, height):
    """Convert a Planck spectrum into an RGBA image with visible-range coloring."""
    mx = np.max(B) if np.max(B) > 0 else 1.
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:,:,:] = 0
    for i in range(width):
        idx = int(i / width * len(wl_nm))
        idx = min(idx, len(wl_nm)-1)
        wl = wl_nm[idx]; val = B[idx] / mx
        if 380 <= wl <= 780:
            r,g,b = wavelength_to_rgb_numba(wl)
            bright = min(1.0, val**0.4)
            rgba[:, i, 0] = int(r*bright)
            rgba[:, i, 1] = int(g*bright)
            rgba[:, i, 2] = int(b*bright)
            rgba[:, i, 3] = 255
        else:
            bright = min(1.0, val**0.4)
            gray = int(bright * 80)
            rgba[:, i, 0] = gray; rgba[:, i, 1] = gray
            rgba[:, i, 2] = int(gray*1.2); rgba[:, i, 3] = 255
    return rgba


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY WIDGETS
# ═══════════════════════════════════════════════════════════════════════

class ImageDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self._data=None; self._pix=None; self.setMinimumSize(350,350)

    def set_rgba(self, arr):
        h,w=arr.shape[:2]; self._data=np.ascontiguousarray(arr)
        img=QImage(self._data.data,w,h,w*4,QImage.Format_RGBA8888)
        self._pix=QPixmap.fromImage(img); self.update()

    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.SmoothPixmapTransform)
        if self._pix:
            sc=self._pix.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
            p.drawPixmap((self.width()-sc.width())//2,(self.height()-sc.height())//2,sc)
        else:
            p.fillRect(self.rect(),QColor(15,15,30))
            p.setPen(QColor(80,80,110)); p.setFont(QFont("Arial",11))
            p.drawText(self.rect(),Qt.AlignCenter,"Computing…")


class ProfileWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data=None; self.color=QColor(80,200,255)
        self.fill_color=QColor(80,200,255,40)
        self.setMinimumHeight(120); self.setMaximumHeight(160)
        self.x_range = None  # (min, max) for labeling

    def set_data(self, d, c=None, fc=None, xr=None):
        self.data=d
        if c: self.color=c
        if fc: self.fill_color=fc
        self.x_range=xr; self.update()

    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height(); p.fillRect(self.rect(),QColor(15,15,30))
        m=40; pw=w-2*m; ph=h-2*m
        p.setPen(QPen(QColor(50,50,75),1))
        p.drawLine(m,m,m,h-m); p.drawLine(m,h-m,w-m,h-m)
        if self.data is None or len(self.data)<2: return
        mx=max(np.max(self.data),1e-10)
        # Fill
        fill=QPainterPath(); fill.moveTo(m,h-m)
        n=len(self.data)
        for i in range(n):
            x=m+i/(n-1)*pw; y=h-m-(self.data[i]/mx)*ph
            fill.lineTo(x,y)
        fill.lineTo(m+pw,h-m); fill.closeSubpath()
        p.fillPath(fill,QBrush(self.fill_color))
        # Line
        path=QPainterPath()
        for i in range(n):
            x=m+i/(n-1)*pw; y=h-m-(self.data[i]/mx)*ph
            (path.moveTo if i==0 else path.lineTo)(x,y)
        p.setPen(QPen(self.color,2)); p.drawPath(path)
        # Labels
        p.setPen(QColor(130,130,160)); p.setFont(QFont("Arial",7))
        p.drawText(m+3,m+11,"I/I₀")
        if self.x_range:
            p.drawText(m,h-m+12,f"{self.x_range[0]:.0f}")
            p.drawText(w-m-30,h-m+12,f"{self.x_range[1]:.0f}")


class SpectrumBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hover_wl=None; self.selected_wl=550.; self.on_select=None
        self.setMinimumHeight(70); self.setMouseTracking(True); self.setCursor(Qt.PointingHandCursor)

    def _wl(self, x):
        m=10; return 380+max(0,min(1,(x-m)/(self.width()-2*m)))*400
    def mouseMoveEvent(self, e): self.hover_wl=self._wl(e.position().x()); self.update()
    def mousePressEvent(self, e):
        self.selected_wl=self._wl(e.position().x()); self.update()
        if self.on_select: self.on_select(self.selected_wl)
    def leaveEvent(self, e): self.hover_wl=None; self.update()

    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        m=10; w=self.width()-2*m; h=self.height()-22
        spec=generate_spectrum_line(int(w))
        for i in range(int(w)):
            p.setPen(QColor(spec[i,0],spec[i,1],spec[i,2])); p.drawLine(m+i,0,m+i,h)
        p.setPen(QPen(QColor(80,80,110),1)); p.drawRect(m,0,int(w),h)
        p.setFont(QFont("Arial",7)); p.setPen(QColor(160,160,180))
        for wl in (400,450,500,550,600,650,700,750):
            x=m+(wl-380)/400*w; p.drawText(int(x)-10,h+12,f"{wl}")
        sx=m+(self.selected_wl-380)/400*w
        p.setPen(QPen(QColor(255,255,80),2)); p.drawLine(int(sx),0,int(sx),h)
        p.setBrush(QColor(255,255,80))
        p.drawPolygon(QPolygonF([QPointF(sx-5,0),QPointF(sx+5,0),QPointF(sx,8)]))
        if self.hover_wl and 380<=self.hover_wl<=780:
            hx=m+(self.hover_wl-380)/400*w
            p.setPen(QPen(Qt.white,1,Qt.DashLine)); p.drawLine(int(hx),0,int(hx),h)
        p.end()


class EMSpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(130)
    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        m=15; w=self.width()-2*m; by=50; bh=35
        regions=[("Radio\nwaves",0.,.30,QColor(60,20,80)),("Micro-\nwaves",.30,.40,QColor(90,20,100)),
                 ("Infrared",.40,.50,QColor(140,30,50)),("Visible",.50,.60,None),
                 ("Ultra-\nviolet",.60,.70,QColor(80,20,140)),("X-rays",.70,.85,QColor(40,50,140)),
                 ("Gamma\nrays",.85,1.,QColor(30,30,90))]
        for name,f0,f1,col in regions:
            x0,x1=m+int(f0*w),m+int(f1*w)
            if col is None:
                spec=generate_spectrum_line(x1-x0)
                for i in range(x1-x0):
                    p.setPen(QColor(spec[i,0],spec[i,1],spec[i,2])); p.drawLine(x0+i,by,x0+i,by+bh)
            else: p.fillRect(x0,by,x1-x0,bh,col)
            p.setPen(QPen(QColor(100,100,130),1)); p.drawRect(x0,by,x1-x0,bh)
            p.setPen(QColor(180,180,200)); p.setFont(QFont("Arial",7)); p.drawText(x0+2,by+bh+18,name)
        p.setPen(QColor(255,255,100)); p.setFont(QFont("Arial",8,QFont.Bold))
        p.drawText(m+int(.50*w),by-5,"← Visible →")
        p.setPen(QPen(QColor(100,100,140),1)); p.drawLine(m,by+bh+40,m+w,by+bh+40)
        p.setFont(QFont("Arial",7))
        p.drawText(m,by+bh+55,"Long λ → Short λ   |   Low f, Low E → High f, High E"); p.end()


class AdditiveMixerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.r_val=self.g_val=self.b_val=255; self.setMinimumSize(280,280)
    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.fillRect(self.rect(),QColor(10,10,20))
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        cx,cy=self.width()/2,self.height()/2; r=min(self.width(),self.height())*.27
        offs=[(0,-r*.55),(-r*.5,r*.3),(r*.5,r*.3)]
        cols=[QColor(self.r_val,0,0),QColor(0,self.g_val,0),QColor(0,0,self.b_val)]
        for(ox,oy),c in zip(offs,cols):
            g=QRadialGradient(cx+ox,cy+oy,r); g.setColorAt(0,c); g.setColorAt(.65,c); g.setColorAt(1,QColor(0,0,0,0))
            p.setBrush(QBrush(g)); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(cx+ox,cy+oy),r,r)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setFont(QFont("Arial",9,QFont.Bold))
        for(ox,oy),lab,lc in zip(offs,"RGB",[QColor(255,150,150),QColor(150,255,150),QColor(150,150,255)]):
            p.setPen(lc); p.drawText(int(cx+ox-5),int(cy+oy+4),lab)
        p.end()


class SubtractiveMixerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.c_val=self.m_val=self.y_val=255; self.setMinimumSize(280,280)
    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.fillRect(self.rect(),Qt.white)
        cx,cy=self.width()/2,self.height()/2; r=min(self.width(),self.height())*.27
        offs=[(0,-r*.55),(-r*.5,r*.3),(r*.5,r*.3)]
        cols=[QColor(0,self.c_val,self.c_val,190),QColor(self.m_val,0,self.m_val,190),QColor(self.y_val,self.y_val,0,190)]
        p.setCompositionMode(QPainter.CompositionMode_Multiply)
        for(ox,oy),c in zip(offs,cols):
            g=QRadialGradient(cx+ox,cy+oy,r); g.setColorAt(0,c); g.setColorAt(.65,c); g.setColorAt(1,QColor(255,255,255,0))
            p.setBrush(QBrush(g)); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(cx+ox,cy+oy),r,r)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setFont(QFont("Arial",9,QFont.Bold))
        for(ox,oy),lab,lc in zip(offs,"CMY",[QColor(0,100,100),QColor(100,0,100),QColor(100,100,0)]):
            p.setPen(lc); p.drawText(int(cx+ox-5),int(cy+oy+4),lab)
        p.end()


class WaveWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wl_px=120; self.amp=80; self.freq=1.; self.phase=0.; self.setMinimumSize(400,200)
        self.timer=QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(30)
    def _tick(self): self.phase+=self.freq*.07; self.update()
    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height(); mid=h/2
        p.fillRect(self.rect(),QColor(15,15,30))
        p.setPen(QPen(QColor(40,40,65),1,Qt.DotLine)); p.drawLine(0,int(mid),w,int(mid))
        if self.wl_px>15:
            p.setPen(QPen(QColor(50,50,75),1,Qt.DashLine))
            x=(self.phase*self.wl_px/(2*math.pi))%self.wl_px
            while x<w: p.drawLine(int(x),0,int(x),h); x+=self.wl_px
        p.setPen(QPen(QColor(80,180,255),3)); path=QPainterPath()
        for x in range(w):
            y=mid-self.amp*math.sin(2*math.pi*x/self.wl_px+self.phase)
            (path.moveTo if x==0 else path.lineTo)(x,y)
        p.drawPath(path)
        p.setPen(QPen(QColor(255,200,80),2)); p.drawLine(25,int(mid),25,int(mid-self.amp))
        p.setFont(QFont("Arial",10,QFont.Bold)); p.drawText(30,int(mid-self.amp/2+4),"A")
        ay=mid+self.amp+25
        if self.wl_px>30:
            p.drawLine(50,int(ay),int(50+self.wl_px),int(ay)); p.drawText(int(50+self.wl_px/2-5),int(ay+16),"λ")
        p.end()


def _draw_arrow(p, x1, y1, x2, y2, col):
    a=math.atan2(y2-y1,x2-x1); sz=12; p.setPen(Qt.NoPen); p.setBrush(col)
    p.drawPolygon(QPolygonF([QPointF(x2,y2),
        QPointF(x2-sz*math.cos(a-.45),y2-sz*math.sin(a-.45)),
        QPointF(x2-sz*math.cos(a+.45),y2-sz*math.sin(a+.45))]))


class ReflectionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.angle=45; self.setMinimumSize(380,300)
    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height(); p.fillRect(self.rect(),QColor(15,15,30))
        my=h*.6; cx=w/2
        p.setPen(QPen(QColor(160,180,210),5)); p.drawLine(20,int(my),w-20,int(my))
        p.setPen(QPen(QColor(70,70,90),1))
        for x in range(20,w-20,14): p.drawLine(x,int(my),x-8,int(my+12))
        p.setPen(QPen(QColor(120,120,140),1,Qt.DashLine)); p.drawLine(int(cx),15,int(cx),int(my))
        a=math.radians(self.angle); rl=min(w,h)*.38
        ix,iy=cx-rl*math.sin(a),my-rl*math.cos(a)
        p.setPen(QPen(QColor(255,220,80),3)); p.drawLine(int(ix),int(iy),int(cx),int(my))
        _draw_arrow(p,ix,iy,cx,my,QColor(255,220,80))
        rx,ry=cx+rl*math.sin(a),my-rl*math.cos(a)
        p.setPen(QPen(QColor(80,200,255),3)); p.drawLine(int(cx),int(my),int(rx),int(ry))
        _draw_arrow(p,cx,my,rx,ry,QColor(80,200,255))
        ar=50
        p.setPen(QPen(QColor(255,220,80,160),2))
        p.drawArc(int(cx-ar),int(my-ar),ar*2,ar*2,90*16,int(self.angle*16))
        p.setPen(QPen(QColor(80,200,255,160),2))
        p.drawArc(int(cx-ar),int(my-ar),ar*2,ar*2,90*16-int(self.angle*16),int(self.angle*16))
        p.setFont(QFont("Arial",11,QFont.Bold))
        p.setPen(QColor(255,220,80)); p.drawText(int(cx-95),int(my-65),f"θᵢ={self.angle}°")
        p.setPen(QColor(80,200,255)); p.drawText(int(cx+20),int(my-65),f"θᵣ={self.angle}°")
        p.end()


class RefractionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.angle=30; self.n1=1.0003; self.n2=1.52; self.setMinimumSize(380,340)
    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height(); my=h/2; cx=w/2
        g1=QLinearGradient(0,0,0,my); g1.setColorAt(0,QColor(18,22,42)); g1.setColorAt(1,QColor(22,30,55))
        p.fillRect(QRectF(0,0,w,my),g1)
        g2=QLinearGradient(0,my,0,h); g2.setColorAt(0,QColor(28,48,78)); g2.setColorAt(1,QColor(18,28,55))
        p.fillRect(QRectF(0,my,w,h-my),g2)
        p.setPen(QPen(QColor(80,120,170),2)); p.drawLine(0,int(my),w,int(my))
        p.setPen(QPen(QColor(120,120,140),1,Qt.DashLine)); p.drawLine(int(cx),15,int(cx),h-15)
        a1=math.radians(self.angle); rl=min(w,h)*.38
        ix,iy=cx-rl*math.sin(a1),my-rl*math.cos(a1)
        p.setPen(QPen(QColor(255,220,80),3)); p.drawLine(int(ix),int(iy),int(cx),int(my))
        _draw_arrow(p,ix,iy,cx,my,QColor(255,220,80))
        sin_a2=self.n1*math.sin(a1)/self.n2; tir=abs(sin_a2)>1
        if not tir:
            a2=math.asin(sin_a2); a2d=math.degrees(a2)
            rx,ry=cx+rl*math.sin(a2),my+rl*math.cos(a2)
            p.setPen(QPen(QColor(80,220,255),3)); p.drawLine(int(cx),int(my),int(rx),int(ry))
            _draw_arrow(p,cx,my,rx,ry,QColor(80,220,255))
            ar=50
            p.setPen(QPen(QColor(255,220,80,160),2))
            p.drawArc(int(cx-ar),int(my-ar),ar*2,ar*2,90*16,int(self.angle*16))
            p.setPen(QPen(QColor(80,220,255,160),2))
            p.drawArc(int(cx-ar),int(my-ar),ar*2,ar*2,-int(a2d*16),int(a2d*16))
            p.setFont(QFont("Arial",11,QFont.Bold)); p.setPen(QColor(80,220,255))
            p.drawText(int(cx+20),int(my+75),f"θ₂={a2d:.1f}°")
        else:
            rx,ry=cx+rl*math.sin(a1),my-rl*math.cos(a1)
            p.setPen(QPen(QColor(255,80,80),3)); p.drawLine(int(cx),int(my),int(rx),int(ry))
            _draw_arrow(p,cx,my,rx,ry,QColor(255,80,80))
            p.setFont(QFont("Arial",12,QFont.Bold)); p.setPen(QColor(255,80,80))
            p.drawText(int(cx-110),int(my+75),"Total Internal Reflection!")
        p.setFont(QFont("Arial",11,QFont.Bold)); p.setPen(QColor(255,220,80))
        p.drawText(int(cx-95),int(my-65),f"θ₁={self.angle}°")
        p.setPen(QColor(160,180,210)); p.setFont(QFont("Arial",10))
        p.drawText(10,22,f"n₁={self.n1:.4f}"); p.drawText(10,int(h-8),f"n₂={self.n2:.4f}")
        if self.n1>self.n2:
            p.setPen(QColor(255,180,80)); p.drawText(10,40,f"Critical={math.degrees(math.asin(self.n2/self.n1)):.1f}°")
        p.end()


class PrismWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(400,340)
    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height(); p.fillRect(self.rect(),QColor(12,12,28))
        cx,cy=w/2,h/2; s=min(w,h)*.38
        top=QPointF(cx,cy-s*.48); bl=QPointF(cx-s*.42,cy+s*.32); br=QPointF(cx+s*.42,cy+s*.32)
        gg=QLinearGradient(bl,br); gg.setColorAt(0,QColor(160,185,220,60)); gg.setColorAt(1,QColor(120,150,200,45))
        p.setBrush(QBrush(gg)); p.setPen(QPen(QColor(170,195,230,140),2))
        p.drawPolygon(QPolygonF([top,bl,br]))
        lm=QPointF((top.x()+bl.x())/2,(top.y()+bl.y())/2)
        p.setPen(QPen(Qt.white,3)); p.drawLine(15,int(lm.y()-20),int(lm.x()),int(lm.y()))
        rm=QPointF((top.x()+br.x())/2,(top.y()+br.y())/2)
        for wl in range(400,710,30):
            n=1.522+4200/wl**2; dev=22+(n-1.522)*650; a=math.radians(dev)
            rl=max(w,h)*.55
            p.setPen(QPen(wavelength_to_rgb(wl),2))
            p.drawLine(int(rm.x()),int(rm.y()),int(rm.x()+rl*math.sin(a)),int(rm.y()+rl*math.cos(a)))
        p.end()


# ═══════════════════════════════════════════════════════════════════════
# NEW WIDGETS — POLARIZATION, LENS, RAYLEIGH, FRESNEL
# ═══════════════════════════════════════════════════════════════════════

class PolarizationWidget(QWidget):
    """Animated polarized light wave with polarizer."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pol_angle = 0       # polarizer angle (degrees)
        self.wave_type = 0       # 0=linear, 1=circular, 2=elliptical
        self.phase = 0.
        self.setMinimumSize(400, 300)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(30)

    def _tick(self):
        self.phase += 0.08; self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(15, 15, 30))

        cx, cy = w / 2, h / 2
        prop_len = w * 0.8
        x0 = w * 0.1

        # Propagation axis
        p.setPen(QPen(QColor(60, 60, 80), 1))
        p.drawLine(int(x0), int(cy), int(x0 + prop_len), int(cy))

        # Polarizer line
        pol_x = x0 + prop_len * 0.5
        pol_rad = math.radians(self.pol_angle)
        plen = 80
        p.setPen(QPen(QColor(200, 200, 60), 3))
        p.drawLine(int(pol_x - plen * math.cos(pol_rad)),
                   int(cy - plen * math.sin(pol_rad)),
                   int(pol_x + plen * math.cos(pol_rad)),
                   int(cy + plen * math.sin(pol_rad)))
        p.setFont(QFont("Arial", 8)); p.setPen(QColor(200, 200, 60))
        p.drawText(int(pol_x - 30), int(cy + plen + 15), f"Polarizer {self.pol_angle}°")

        # Draw wave before polarizer (unpolarized → show two components)
        amp = 60
        # Vertical component (blue)
        p.setPen(QPen(QColor(100, 150, 255), 2))
        path_v = QPainterPath()
        for i in range(int(prop_len * 0.45)):
            xp = x0 + i
            yv = cy - amp * math.sin(2 * math.pi * i / 80 + self.phase)
            (path_v.moveTo if i == 0 else path_v.lineTo)(xp, yv)
        p.drawPath(path_v)

        # Horizontal component (red) — shown as depth using oblique projection
        p.setPen(QPen(QColor(255, 120, 100), 2))
        path_h = QPainterPath()
        for i in range(int(prop_len * 0.45)):
            xp = x0 + i
            xh = xp + amp * 0.4 * math.sin(2 * math.pi * i / 80 + self.phase + math.pi/3)
            (path_h.moveTo if i == 0 else path_h.lineTo)(xh, cy + amp * 0.2)
        p.drawPath(path_h)

        # After polarizer — transmitted component
        trans_amp = amp * abs(math.cos(pol_rad))  # Malus's law amplitude
        if self.wave_type == 1:  # circular
            trans_amp = amp * 0.707
        p.setPen(QPen(QColor(80, 255, 120), 3))
        path_t = QPainterPath()
        for i in range(int(prop_len * 0.45)):
            xp = pol_x + i
            # Transmitted oscillation along polarizer direction
            osc = trans_amp * math.sin(2 * math.pi * i / 80 + self.phase)
            yp = cy - osc * math.sin(pol_rad)
            xo = osc * math.cos(pol_rad) * 0.4  # oblique projection
            (path_t.moveTo if i == 0 else path_t.lineTo)(xp + xo, yp)
        p.drawPath(path_t)

        # Labels
        p.setFont(QFont("Arial", 9))
        p.setPen(QColor(100, 150, 255)); p.drawText(int(x0), 20, "Vertical component")
        p.setPen(QColor(255, 120, 100)); p.drawText(int(x0), 35, "Horizontal component")
        p.setPen(QColor(80, 255, 120)); p.drawText(int(pol_x + 10), 20, "Transmitted")

        # Malus's law
        intensity = math.cos(pol_rad) ** 2
        p.setFont(QFont("Arial", 11, QFont.Bold)); p.setPen(QColor(220, 220, 240))
        p.drawText(10, h - 15, f"Malus's Law: I = I₀·cos²({self.pol_angle}°) = {intensity:.3f}·I₀")
        p.end()


class LensWidget(QWidget):
    """Interactive thin lens ray diagram."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.focal_len = 120    # pixels (positive = converging)
        self.obj_dist = 250     # pixels (positive = left of lens)
        self.obj_height = 60    # pixels
        self.lens_type = 0      # 0=converging, 1=diverging
        self.setMinimumSize(500, 350)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); p.fillRect(self.rect(), QColor(15, 15, 30))
        cx, cy = w / 2, h / 2

        f = self.focal_len if self.lens_type == 0 else -self.focal_len
        do = self.obj_dist  # object distance (positive = left)

        # Optical axis
        p.setPen(QPen(QColor(50, 50, 70), 1, Qt.DashLine))
        p.drawLine(0, int(cy), w, int(cy))

        # Lens
        lens_h = 130
        if self.lens_type == 0:
            # Converging lens (biconvex shape)
            p.setPen(QPen(QColor(120, 180, 255), 3))
            path = QPainterPath()
            path.moveTo(int(cx), int(cy - lens_h))
            path.quadTo(int(cx + 15), int(cy), int(cx), int(cy + lens_h))
            path.quadTo(int(cx - 15), int(cy), int(cx), int(cy - lens_h))
            p.drawPath(path)
        else:
            # Diverging lens (biconcave)
            p.setPen(QPen(QColor(255, 150, 100), 3))
            path = QPainterPath()
            path.moveTo(int(cx + 8), int(cy - lens_h))
            path.quadTo(int(cx - 8), int(cy), int(cx + 8), int(cy + lens_h))
            path.moveTo(int(cx - 8), int(cy - lens_h))
            path.quadTo(int(cx + 8), int(cy), int(cx - 8), int(cy + lens_h))
            p.drawPath(path)

        # Focal points
        p.setPen(QPen(QColor(255, 200, 80), 2))
        p.setBrush(QColor(255, 200, 80))
        for fx in [cx - abs(f), cx + abs(f)]:
            p.drawEllipse(QPointF(fx, cy), 4, 4)
        p.setFont(QFont("Arial", 9)); p.setPen(QColor(255, 200, 80))
        p.drawText(int(cx - abs(f) - 5), int(cy + 18), "F")
        p.drawText(int(cx + abs(f) - 5), int(cy + 18), "F'")

        # Object (arrow on left)
        obj_x = cx - do
        p.setPen(QPen(QColor(80, 255, 120), 3))
        p.drawLine(int(obj_x), int(cy), int(obj_x), int(cy - self.obj_height))
        _draw_arrow(p, obj_x, cy, obj_x, cy - self.obj_height, QColor(80, 255, 120))
        p.setFont(QFont("Arial", 9)); p.setPen(QColor(80, 255, 120))
        p.drawText(int(obj_x - 5), int(cy + 15), "Object")

        # Thin lens equation: 1/f = 1/do + 1/di
        if abs(do - f) > 1:
            di = f * do / (do - f)   # image distance
            m = -di / do              # magnification
            img_h = m * self.obj_height
            img_x = cx + di

            # Ray 1: Parallel to axis → through F'
            p.setPen(QPen(QColor(255, 220, 80), 2))
            p.drawLine(int(obj_x), int(cy - self.obj_height), int(cx), int(cy - self.obj_height))
            if self.lens_type == 0:
                p.drawLine(int(cx), int(cy - self.obj_height), int(img_x), int(cy - img_h))
            else:
                # Virtual ray extension
                p.drawLine(int(cx), int(cy - self.obj_height), int(cx + 200), int(cy - self.obj_height + 200 * self.obj_height / abs(f)))
                p.setPen(QPen(QColor(255, 220, 80, 80), 1, Qt.DashLine))
                p.drawLine(int(cx), int(cy - self.obj_height), int(img_x), int(cy - img_h))

            # Ray 2: Through center (undeviated)
            p.setPen(QPen(QColor(80, 200, 255), 2))
            p.drawLine(int(obj_x), int(cy - self.obj_height), int(min(img_x if di > 0 else cx + 200, w - 10)),
                       int(cy - self.obj_height + (min(img_x if di > 0 else cx + 200, w - 10) - obj_x) * self.obj_height / do))

            # Ray 3: Through F → parallel after lens
            if self.lens_type == 0:
                p.setPen(QPen(QColor(255, 130, 80), 2))
                y_at_lens = cy - self.obj_height * (do - abs(f)) / abs(f) if abs(f) > 1 else cy
                p.drawLine(int(obj_x), int(cy - self.obj_height), int(cx), int(y_at_lens))
                p.drawLine(int(cx), int(y_at_lens), int(min(img_x, w - 10)), int(y_at_lens))

            # Image arrow
            if abs(img_x - cx) < w:
                if di > 0:
                    p.setPen(QPen(QColor(255, 100, 100), 3))
                else:
                    p.setPen(QPen(QColor(255, 100, 100, 120), 2, Qt.DashLine))
                p.drawLine(int(img_x), int(cy), int(img_x), int(cy - img_h))
                _draw_arrow(p, img_x, cy, img_x, cy - img_h, QColor(255, 100, 100))
                p.setFont(QFont("Arial", 9)); p.setPen(QColor(255, 100, 100))
                label = "Real Image" if di > 0 else "Virtual Image"
                p.drawText(int(img_x - 20), int(cy + 15), label)
        else:
            di = None; m = None; img_h = None

        p.end()


class RayleighSkyWidget(QWidget):
    """Simulates Rayleigh scattering — why the sky is blue and sunsets are red."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sun_elevation = 45  # degrees (90=zenith, 0=horizon)
        self.path_length = 1.0   # atmospheric path multiplier
        self.setMinimumSize(400, 250)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Compute sky color based on Rayleigh scattering
        # At zenith: short path → blue dominates
        # At horizon: long path → blue scattered away → red/orange
        elev = max(1, self.sun_elevation)
        # Atmospheric path length ∝ 1/sin(elevation)
        path = self.path_length / math.sin(math.radians(elev))

        # Scattering probability ∝ 1/λ⁴
        # Transmitted light: I = I₀ × exp(-τ/λ⁴) where τ ∝ path length
        tau_base = 0.05 * path  # optical depth scale

        # Compute transmitted color (what reaches observer directly)
        vis_wls = np.linspace(380, 780, 80)
        r_sum, g_sum, b_sum = 0., 0., 0.
        for wl in vis_wls:
            tau = tau_base * (550. / wl) ** 4
            transmission = math.exp(-tau)
            rgb = wavelength_to_rgb_numba(wl)
            r_sum += rgb[0] * transmission
            g_sum += rgb[1] * transmission
            b_sum += rgb[2] * transmission

        mx = max(r_sum, g_sum, b_sum, 1.)
        sun_r, sun_g, sun_b = int(min(255, r_sum/mx*255)), int(min(255, g_sum/mx*255)), int(min(255, b_sum/mx*255))
        sun_color = QColor(sun_r, sun_g, sun_b)

        # Sky color = scattered light (complementary to transmitted)
        # Scattered intensity ∝ 1/λ⁴, so sky is blue
        scat_wls = np.linspace(380, 780, 80)
        sr, sg, sb = 0., 0., 0.
        for wl in scat_wls:
            scat = (550. / wl) ** 4 * math.exp(-tau_base * 0.3 * (550./wl)**4)
            rgb = wavelength_to_rgb_numba(wl)
            sr += rgb[0] * scat; sg += rgb[1] * scat; sb += rgb[2] * scat
        smx = max(sr, sg, sb, 1.)
        sky_color = QColor(int(min(255, sr/smx*180)), int(min(255, sg/smx*180)), int(min(255, sb/smx*200)))

        # Draw sky gradient
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, sky_color.darker(120))
        grad.setColorAt(0.7, sky_color)
        grad.setColorAt(1, sky_color.lighter(130))
        p.fillRect(self.rect(), grad)

        # Draw sun
        sun_y = h - h * elev / 90.
        sun_x = w * 0.5
        sg2 = QRadialGradient(sun_x, sun_y, 40)
        sg2.setColorAt(0, sun_color.lighter(150))
        sg2.setColorAt(0.3, sun_color)
        sg2.setColorAt(1, QColor(sun_color.red(), sun_color.green(), sun_color.blue(), 0))
        p.setBrush(QBrush(sg2)); p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(sun_x, sun_y), 40, 40)

        # Ground
        p.fillRect(QRectF(0, h - 20, w, 20), QColor(40, 60, 30))
        p.end()


class FresnelWidget(QWidget):
    """Fresnel equation visualization showing Rs, Rp vs angle."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.n1 = 1.0; self.n2 = 1.5
        self.setMinimumSize(400, 280)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(15, 15, 30))

        m = 50; pw = w - 2*m; ph = h - 2*m

        # Axes
        p.setPen(QPen(QColor(60, 60, 80), 1))
        p.drawLine(m, m, m, h-m); p.drawLine(m, h-m, w-m, h-m)

        # Compute Fresnel coefficients
        angles = np.linspace(0, 89, 200)
        Rs, Rp, Ts, Tp = fresnel_coefficients_numba(angles, self.n1, self.n2)

        # Draw curves
        def draw_curve(data, color, width=2):
            path = QPainterPath()
            for i, val in enumerate(data):
                x = m + i / (len(data)-1) * pw
                y = h - m - val * ph
                (path.moveTo if i == 0 else path.lineTo)(x, y)
            p.setPen(QPen(color, width)); p.drawPath(path)

        draw_curve(Rs, QColor(255, 120, 100), 2)
        draw_curve(Rp, QColor(100, 180, 255), 2)
        draw_curve(Ts, QColor(255, 120, 100, 120), 1)
        draw_curve(Tp, QColor(100, 180, 255, 120), 1)

        # Brewster angle marker
        if self.n1 > 0:
            brewster = math.degrees(math.atan(self.n2 / self.n1))
            bx = m + brewster / 89 * pw
            p.setPen(QPen(QColor(255, 255, 100), 1, Qt.DashLine))
            p.drawLine(int(bx), m, int(bx), h-m)
            p.setFont(QFont("Arial", 8)); p.setPen(QColor(255, 255, 100))
            p.drawText(int(bx + 3), m + 12, f"θ_B={brewster:.1f}°")

        # Critical angle marker (if n1 > n2)
        if self.n1 > self.n2:
            critical = math.degrees(math.asin(self.n2 / self.n1))
            crx = m + critical / 89 * pw
            p.setPen(QPen(QColor(255, 80, 80), 1, Qt.DashLine))
            p.drawLine(int(crx), m, int(crx), h-m)
            p.setFont(QFont("Arial", 8)); p.setPen(QColor(255, 80, 80))
            p.drawText(int(crx + 3), m + 25, f"θ_c={critical:.1f}°")

        # Labels
        p.setFont(QFont("Arial", 8)); p.setPen(QColor(130, 130, 160))
        p.drawText(m, h-m+14, "0°"); p.drawText(w-m-15, h-m+14, "89°")
        p.drawText(m-35, m+5, "1.0"); p.drawText(m-25, h-m+5, "0")
        p.setFont(QFont("Arial", 9, QFont.Bold))
        p.setPen(QColor(255, 120, 100)); p.drawText(w-120, m+5, "Rs / Ts")
        p.setPen(QColor(100, 180, 255)); p.drawText(w-120, m+18, "Rp / Tp")
        p.end()


# ═══════════════════════════════════════════════════════════════════════
# PAGE WIDGETS
# ═══════════════════════════════════════════════════════════════════════

class SpectrumPage(QWidget):
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(12)
        ly.addWidget(heading("🌈 Electromagnetic Spectrum"))
        ly.addWidget(body("The EM spectrum spans from radio waves (meters) to gamma rays (pm). "
                          "Visible light (380–780 nm) is a tiny slice. Click the bar to explore."))
        self.em=EMSpectrumWidget(); ly.addWidget(self.em)
        ly.addWidget(heading("Visible Spectrum Explorer",14))
        self.bar=SpectrumBar(); self.bar.on_select=self._sel; ly.addWidget(self.bar)
        self.info=info_label("Click on the visible spectrum above to explore wavelengths.")
        ly.addWidget(self.info); ly.addStretch()

    def _sel(self, wl):
        f=wl_to_freq(wl); e=wl_to_energy(wl); rgb=wavelength_to_rgb(wl)
        self.info.setText(
            f"🔬 λ={wl:.1f} nm  |  {color_name(wl)}\n"
            f"📡 f={f:.3e} Hz  ({f/1e12:.1f} THz)\n"
            f"⚡ E={e:.3f} eV  ({e*EV:.3e} J)\n"
            f"🎨 RGB=({rgb.red()},{rgb.green()},{rgb.blue()})\n"
            f"📐 c=λf → {C:.3e} m/s = {wl*1e-9:.3e} m × {f:.3e} Hz\n"
            f"📂 Region: {em_region(wl)}")


class ColorMixingPage(QWidget):
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(10)
        ly.addWidget(heading("🎨 Color Mixing"))
        ly.addWidget(body("Additive=light (RGB→White). Subtractive=pigment (CMY→Black)."))
        g=QGridLayout()
        g.addWidget(heading("Additive (Light)",13),0,0)
        self.add_mix=AdditiveMixerWidget(); g.addWidget(self.add_mix,1,0)
        ac=QVBoxLayout(); self.add_sl={}
        for ch,col,attr in [("R","#ff6666","r_val"),("G","#66ff66","g_val"),("B","#6666ff","b_val")]:
            l=QLabel(ch); l.setStyleSheet(f"color:{col};font-weight:bold;font-size:14px;"); ac.addWidget(l)
            s=make_slider(0,255,255,lambda v,a=attr:self._set_add(a,v)); ac.addWidget(s); self.add_sl[ch]=s
        self.add_info=info_label("R=255 G=255 B=255 → White"); ac.addWidget(self.add_info)
        g.addLayout(ac,1,1)
        g.addWidget(heading("Subtractive (Pigment)",13),0,2)
        self.sub_mix=SubtractiveMixerWidget(); g.addWidget(self.sub_mix,1,2)
        sc=QVBoxLayout(); self.sub_sl={}
        for ch,col,attr in [("C","#00cccc","c_val"),("M","#cc00cc","m_val"),("Y","#cccc00","y_val")]:
            l=QLabel(ch); l.setStyleSheet(f"color:{col};font-weight:bold;font-size:14px;"); sc.addWidget(l)
            s=make_slider(0,255,255,lambda v,a=attr:self._set_sub(a,v)); sc.addWidget(s); self.sub_sl[ch]=s
        self.sub_info=info_label("C=255 M=255 Y=255 → Black"); sc.addWidget(self.sub_info)
        g.addLayout(sc,1,3); ly.addLayout(g); ly.addStretch()

    def _set_add(self,attr,v):
        setattr(self.add_mix,attr,v); self.add_mix.update()
        m=self.add_mix; self.add_info.setText(f"R={m.r_val} G={m.g_val} B={m.b_val}\nResult: {QColor(m.r_val,m.g_val,m.b_val).name()}")
    def _set_sub(self,attr,v):
        setattr(self.sub_mix,attr,v); self.sub_mix.update()
        m=self.sub_mix; r=255-min(255,m.c_val); g=255-min(255,m.m_val); b=255-min(255,m.y_val)
        self.sub_info.setText(f"C={m.c_val} M={m.m_val} Y={m.y_val}\n≈ {QColor(r,g,b).name()}")


class WavePage(QWidget):
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(10)
        ly.addWidget(heading("🌊 Wave Properties of Light"))
        ly.addWidget(body("λ=wavelength, f=frequency, A=amplitude, c=λf ≈ 3×10⁸ m/s. Amplitude ∝ √Intensity."))
        self.wave=WaveWidget(); ly.addWidget(self.wave)
        g=QGridLayout()
        g.addWidget(QLabel("Wavelength (px):"),0,0)
        self.sl_wl=make_slider(30,300,120,self._u); g.addWidget(self.sl_wl,0,1); self.lw=QLabel("120"); g.addWidget(self.lw,0,2)
        g.addWidget(QLabel("Amplitude:"),1,0)
        self.sl_a=make_slider(10,120,80,self._u); g.addWidget(self.sl_a,1,1); self.la=QLabel("80"); g.addWidget(self.la,1,2)
        g.addWidget(QLabel("Frequency:"),2,0)
        self.sl_f=make_slider(1,50,10,self._u); g.addWidget(self.sl_f,2,1); self.lf=QLabel("1.0"); g.addWidget(self.lf,2,2)
        ly.addLayout(g); self.info=info_label(); ly.addWidget(self.info); self._u(); ly.addStretch()

    def _u(self):
        wl=self.sl_wl.value(); amp=self.sl_a.value(); freq=self.sl_f.value()/10
        self.wave.wl_px=wl; self.wave.amp=amp; self.wave.freq=freq
        self.lw.setText(str(wl)); self.la.setText(str(amp)); self.lf.setText(f"{freq:.1f}")
        self.info.setText(f"λ={wl}px | A={amp}px | f={freq:.1f}Hz | T={1/freq:.2f}s\n"
                          f"Physical: λ=550nm → f={wl_to_freq(550):.3e}Hz, E={wl_to_energy(550):.3f}eV")


class ReflectionPage(QWidget):
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(10)
        ly.addWidget(heading("🪞 Reflection"))
        ly.addWidget(body("Law of Reflection: θᵢ = θᵣ (measured from the normal). "
                          "Specular=smooth surface, Diffuse=rough surface."))
        self.cv=ReflectionWidget(); ly.addWidget(self.cv)
        hl=QHBoxLayout(); hl.addWidget(QLabel("Angle:"))
        self.sl=make_slider(5,85,45,self._u); hl.addWidget(self.sl)
        self.lb=QLabel("45°"); hl.addWidget(self.lb); ly.addLayout(hl)
        self.info=info_label(); ly.addWidget(self.info); self._u(); ly.addStretch()

    def _u(self):
        a=self.sl.value(); self.cv.angle=a; self.cv.update(); self.lb.setText(f"{a}°")
        self.info.setText(f"θᵢ={a}° | θᵣ={a}° ✓  Law of Reflection satisfied")


class RefractionPage(QWidget):
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(10)
        ly.addWidget(heading("🔀 Refraction — Snell's Law"))
        ly.addWidget(body("n₁ sin θ₁ = n₂ sin θ₂. Light bends toward normal in denser medium. "
                          "If n₁>n₂ and θ>critical angle → Total Internal Reflection."))
        self.cv=RefractionWidget(); ly.addWidget(self.cv)
        g=QGridLayout()
        g.addWidget(QLabel("θ₁:"),0,0); self.sl_a=make_slider(0,89,30,self._u); g.addWidget(self.sl_a,0,1)
        self.la=QLabel("30°"); g.addWidget(self.la,0,2)
        g.addWidget(QLabel("Medium 1:"),1,0); self.cb1=QComboBox(); self.cb1.addItems(MEDIA.keys())
        self.cb1.setCurrentText("Air"); self.cb1.currentTextChanged.connect(self._u); g.addWidget(self.cb1,1,1)
        g.addWidget(QLabel("Medium 2:"),2,0); self.cb2=QComboBox(); self.cb2.addItems(MEDIA.keys())
        self.cb2.setCurrentText("Glass (Crown)"); self.cb2.currentTextChanged.connect(self._u); g.addWidget(self.cb2,2,1)
        ly.addLayout(g); self.info=info_label(); ly.addWidget(self.info); self._u(); ly.addStretch()

    def _u(self):
        a=self.sl_a.value(); n1=MEDIA.get(self.cb1.currentText(),1.); n2=MEDIA.get(self.cb2.currentText(),1.)
        self.la.setText(f"{a}°"); self.cv.angle=a; self.cv.n1=n1; self.cv.n2=n2; self.cv.update()
        s2=n1*math.sin(math.radians(a))/n2; tir=abs(s2)>1
        a2=math.degrees(math.asin(s2)) if not tir else None
        ca=f"\nCritical angle: {math.degrees(math.asin(n2/n1)):.1f}°" if n1>n2 else ""
        v2=C/n2
        self.info.setText(
            f"n₁={n1:.4f} | n₂={n2:.4f}\n"
            f"Snell: {n1:.4f}×sin({a}°) = {n2:.4f}×sin(θ₂)\n"
            f"θ₂ = {f'{a2:.1f}°' if a2 else 'N/A'}\n"
            f"{'⚡ TIR!' if tir else ''}{ca}\n"
            f"Speed in medium 2: v = c/n = {v2:.3e} m/s")


class DispersionPage(QWidget):
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(10)
        ly.addWidget(heading("🔬 Dispersion"))
        ly.addWidget(body("Dispersion: n(λ) varies with wavelength (Cauchy: n=A+B/λ²). "
                          "Shorter λ → higher n → more bending. A prism separates white light."))
        self.prism=PrismWidget(); ly.addWidget(self.prism)
        lines=["Cauchy: n(λ) = 1.522 + 4200/λ²\n"]
        for wl in [400,450,500,550,600,650,700]:
            lines.append(f"  λ={wl}nm → n={1.522+4200/wl**2:.4f}  ({color_name(wl)})")
        self.info=info_label("\n".join(lines)); ly.addWidget(self.info); ly.addStretch()


class DiffractionPage(QWidget):
    PATTERNS=["Single Slit","Double Slit","Diffraction Grating"]
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(8)
        ly.addWidget(heading("🔬 Diffraction & Interference  [GPU]"))
        ly.addWidget(body("Single: I=I₀·sinc²(β). Double: I=I₀·sinc²(β)·cos²(δ). "
                          "Grating: I=I₀·sinc²(β)·[sin(Nδ)/(Nsinδ)]². Computed on GPU with CuPy."))
        top=QHBoxLayout(); self.img=ImageDisplay(); top.addWidget(self.img,3)
        right=QVBoxLayout(); right.addWidget(heading("1D Profile",12)); self.prof=ProfileWidget(); right.addWidget(self.prof)
        ctrl=QGroupBox("Parameters"); cl=QVBoxLayout(ctrl)
        cl.addWidget(QLabel("Pattern:")); self.cb_pat=QComboBox(); self.cb_pat.addItems(self.PATTERNS)
        self.cb_pat.currentIndexChanged.connect(self._compute); cl.addWidget(self.cb_pat)
        for lbl,lo,hi,val,attr in [("Wavelength (nm):",380,780,550,"sl_wl"),("Slit width (μm):",1,100,10,"sl_a"),
                                    ("Slit separation (μm):",5,200,40,"sl_d"),("Number of slits:",2,20,5,"sl_n")]:
            cl.addWidget(QLabel(lbl)); s=make_slider(lo,hi,val,self._compute); setattr(self,attr,s); cl.addWidget(s)
        self.lbl_be=QLabel(); self.lbl_be.setStyleSheet("color:#60d0a0;font-size:11px;font-weight:bold;"); cl.addWidget(self.lbl_be)
        self.info=info_label(); cl.addWidget(self.info); right.addWidget(ctrl); top.addLayout(right,2); ly.addLayout(top); ly.addStretch()
        QTimer.singleShot(100,self._compute)

    def _compute(self):
        pat=self.cb_pat.currentIndex(); wl=self.sl_wl.value(); a=self.sl_a.value(); d=self.sl_d.value(); N=self.sl_n.value()
        t0=time.perf_counter()
        I2d=gpu_compute_diffraction_2d(pat,wl,a,d,N,res=400)
        lam=wl*1e-9; a_m=a*1e-6; d_m=d*1e-6
        y_coords=np.linspace(-0.03,0.03,1000)
        I1d=generate_diffraction_1d_numba(y_coords,lam,a_m,d_m,N if pat==2 else (1 if pat==0 else 2))
        dt=(time.perf_counter()-t0)*1000
        rgb=wavelength_to_rgb_numba(float(wl)); rgba=intensity_to_rgba(I2d,rgb,gamma=0.38)
        self.img.set_rgba(rgba); self.prof.set_data(I1d,wavelength_to_rgb(wl),xr=(-30,30))
        be="🟢 CuPy GPU" if HAS_CUPY else "🔵 NumPy CPU"; self.lbl_be.setText(f"{be}  |  {dt:.1f} ms")
        formulas={0:"I=I₀·[sin(β)/β]²",1:"I=I₀·[sin(β)/β]²·cos²(δ)",2:f"I=I₀·[sin(β)/β]²·[sin({N}δ)/({N}sinδ)]²"}
        self.info.setText(f"{self.PATTERNS[pat]} | λ={wl}nm, a={a}μm, d={d}μm, N={N}\n{formulas[pat]}\nCompute: {dt:.1f}ms ({be})")


class WaveSimPage(QWidget):
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(8)
        ly.addWidget(heading("🌊 2D Wave Simulation  [GPU]"))
        ly.addWidget(body("Click the field to place coherent point sources (up to 8). "
                          "Watch constructive (bright) and destructive (dark) interference. ψ=Σsin(2πr/λ-ωt)/√r. GPU: CuPy."))
        top=QHBoxLayout(); self.img=ImageDisplay(); self.img.setMinimumSize(400,400)
        self.img.mousePressEvent=self._click; top.addWidget(self.img,3)
        right=QVBoxLayout(); ctrl=QGroupBox("Controls"); cl=QVBoxLayout(ctrl)
        cl.addWidget(QLabel("Wavelength (units):")); self.sl_wl=make_slider(5,40,15,self._param); cl.addWidget(self.sl_wl)
        self.btn_clear=QPushButton("Clear Sources")
        self.btn_clear.setStyleSheet("background:#602020;color:white;border-radius:5px;padding:8px;")
        self.btn_clear.clicked.connect(self._clear); cl.addWidget(self.btn_clear)
        self.btn_pause=QPushButton("⏸ Pause")
        self.btn_pause.setStyleSheet("background:#204060;color:white;border-radius:5px;padding:8px;")
        self.btn_pause.clicked.connect(self._toggle); cl.addWidget(self.btn_pause)
        self.lbl_be=QLabel(); self.lbl_be.setStyleSheet("color:#60d0a0;font-size:11px;font-weight:bold;"); cl.addWidget(self.lbl_be)
        self.info=info_label("Click to place sources (up to 8)"); cl.addWidget(self.info); right.addWidget(ctrl)
        top.addLayout(right,1); ly.addLayout(top); ly.addStretch()
        self.sources=[]; self.phase=0.; self.paused=False
        self.timer=QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(33)

    def _click(self,e):
        if len(self.sources)>=8: return
        w=self.img.width(); h=self.img.height(); sz=min(w,h)
        ox=(w-sz)/2; oy=(h-sz)/2
        px=e.position().x(); py=e.position().y()
        ux=(px-ox)/sz*20-10; uy=(py-oy)/sz*20-10
        if -10<=ux<=10 and -10<=uy<=10:
            self.sources.append((ux,uy))
            self.info.setText(f"Sources: {len(self.sources)} | "+", ".join(f"({s[0]:.1f},{s[1]:.1f})" for s in self.sources))
    def _clear(self): self.sources.clear(); self.info.setText("Click to place sources (up to 8)")
    def _toggle(self): self.paused=not self.paused; self.btn_pause.setText("▶ Play" if self.paused else "⏸ Pause")
    def _param(self): pass

    def _tick(self):
        if self.paused or not self.sources: return
        self.phase+=0.15; t0=time.perf_counter()
        field=gpu_compute_wave_field_2d(self.sources,self.phase,res=350)
        dt=(time.perf_counter()-t0)*1000; rgba=wave_to_rgba(field); self.img.set_rgba(rgba)
        be="🟢 CuPy GPU" if HAS_CUPY else "🔵 NumPy CPU"; self.lbl_be.setText(f"{be} | {dt:.1f} ms/frame")


# ─── NEW FEATURE PAGES ────────────────────────────────────────────────

class BlackbodyPage(QWidget):
    """Blackbody radiation: Planck's law with temperature slider."""
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(8)
        ly.addWidget(heading("🌡️ Blackbody Radiation  [GPU]"))
        ly.addWidget(body(
            "<b>Planck's Law:</b> B(λ,T) = 2hc²/λ⁵ × 1/(e^(hc/λkT)−1) describes the spectral "
            "radiance of a perfect emitter at temperature T.\n\n"
            "<b>Wien's Law:</b> λ_max = b/T  (b = 2.898×10⁻³ m·K) — peak wavelength shifts blue as T increases.\n"
            "<b>Stefan-Boltzmann:</b> P = σT⁴ — total radiated power per unit area."))

        self.prof = ProfileWidget(); self.prof.setMinimumHeight(180); ly.addWidget(self.prof)

        # Temperature slider
        g = QGridLayout()
        g.addWidget(QLabel("Temperature (K):"), 0, 0)
        self.sl_T = make_slider(500, 15000, 5778, self._compute)
        g.addWidget(self.sl_T, 0, 1)
        self.lbl_T = QLabel("5778 K"); self.lbl_T.setStyleSheet("font-size:14px;font-weight:bold;color:#ffcc44;")
        g.addWidget(self.lbl_T, 0, 2)

        self.lbl_be=QLabel(); self.lbl_be.setStyleSheet("color:#60d0a0;font-size:11px;font-weight:bold;")
        g.addWidget(self.lbl_be, 1, 0, 1, 3)
        ly.addLayout(g)

        self.info = info_label(); ly.addWidget(self.info)
        self.color_preview = QLabel(); self.color_preview.setFixedHeight(30)
        self.color_preview.setStyleSheet("border-radius:6px;"); ly.addWidget(self.color_preview)
        ly.addStretch()
        QTimer.singleShot(100, self._compute)

    def _compute(self):
        T = self.sl_T.value()
        self.lbl_T.setText(f"{T} K")
        t0 = time.perf_counter()

        # Wavelength array: 100nm to 3000nm
        wl_m = np.linspace(100e-9, 3000e-9, 1000)
        wl_nm = wl_m * 1e9

        # Planck spectrum (Numba-accelerated)
        B = planck_spectrum_numba(wl_m, T)

        dt = (time.perf_counter() - t0) * 1000

        # Normalize for display
        mx = np.max(B) if np.max(B) > 0 else 1.
        B_norm = B / mx

        # Color the profile curve by wavelength
        self.prof.set_data(B_norm, QColor(255, 200, 80),
                           QColor(255, 200, 80, 40), xr=(100, 3000))

        # Wien's law
        wien_nm = WIEN_B / T * 1e9
        # Stefan-Boltzmann
        P = STEFAN * T**4
        # Peak spectral radiance
        B_peak = np.max(B)

        # Blackbody color
        bb_color = blackbody_rgb(T)
        self.color_preview.setStyleSheet(
            f"background-color:{bb_color.name()};border-radius:6px;border:1px solid #444;")

        region = "Visible" if 380 <= wien_nm <= 780 else ("UV" if wien_nm < 380 else "Infrared")

        self.info.setText(
            f"🌡️ T = {T} K\n"
            f"📊 Wien's Law: λ_peak = {wien_nm:.1f} nm  ({region}: {color_name(wien_nm) if 380<=wien_nm<=780 else region})\n"
            f"🔥 Stefan-Boltzmann: P = σT⁴ = {P:.3e} W/m²\n"
            f"📈 Peak radiance: B_peak = {B_peak:.3e} W·sr⁻¹·m⁻³\n"
            f"⚡ Luminosity relative to Sun (5778K): {(T/5778)**4:.3f}×\n"
            f"⏱ Compute: {dt:.1f} ms")

        be = "🟢 CuPy GPU" if HAS_CUPY else "🔵 NumPy CPU + Numba"
        self.lbl_be.setText(f"{be}  |  {dt:.1f} ms")

        # Override profile paint to add visible-range shading
        self.prof.update()


class PolarizationPage(QWidget):
    """Polarization of light with interactive polarizer."""
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(8)
        ly.addWidget(heading("↕️ Polarization"))
        ly.addWidget(body(
            "Light is a transverse EM wave. <b>Polarization</b> describes the orientation of the "
            "electric field oscillation. <b>Malus's Law:</b> I = I₀·cos²θ — transmitted intensity "
            "through a polarizer at angle θ to the polarization direction."))

        self.cv = PolarizationWidget(); ly.addWidget(self.cv)

        g = QGridLayout()
        g.addWidget(QLabel("Polarizer Angle:"), 0, 0)
        self.sl_pol = make_slider(0, 90, 0, self._upd)
        g.addWidget(self.sl_pol, 0, 1)
        self.lbl_pol = QLabel("0°"); self.lbl_pol.setStyleSheet("font-weight:bold;color:#cccc44;")
        g.addWidget(self.lbl_pol, 0, 2)

        g.addWidget(QLabel("Wave Type:"), 1, 0)
        self.cb_type = QComboBox(); self.cb_type.addItems(["Linear", "Circular", "Elliptical"])
        self.cb_type.currentIndexChanged.connect(self._upd); g.addWidget(self.cb_type, 1, 1)
        ly.addLayout(g)

        self.info = info_label(); ly.addWidget(self.info); self._upd(); ly.addStretch()

    def _upd(self):
        a = self.sl_pol.value(); wt = self.cb_type.currentIndex()
        self.cv.pol_angle = a; self.cv.wave_type = wt; self.cv.update()
        self.lbl_pol.setText(f"{a}°")
        intensity = math.cos(math.radians(a))**2
        self.info.setText(
            f"↕️ Polarizer angle: {a}°\n"
            f"📐 Malus's Law: I = I₀·cos²({a}°) = {intensity:.4f}·I₀\n"
            f"💡 Transmitted: {intensity*100:.1f}% of incident intensity\n"
            f"🔄 Wave type: {['Linear','Circular','Elliptical'][wt]}\n\n"
            f"Polarization types:\n"
            f"  • Linear: E oscillates in one plane\n"
            f"  • Circular: E rotates with constant amplitude\n"
            f"  • Elliptical: E traces an ellipse")


class LensPage(QWidget):
    """Thin lens ray tracing simulation."""
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(8)
        ly.addWidget(heading("🔭 Thin Lens Simulator"))
        ly.addWidget(body(
            "<b>Thin Lens Equation:</b> 1/f = 1/dₒ + 1/dᵢ\n"
            "<b>Magnification:</b> m = −dᵢ/dₒ\n"
            "Converging lens (f>0): real + inverted image when dₒ>f. "
            "Diverging lens (f<0): always virtual + upright image."))

        self.cv = LensWidget(); ly.addWidget(self.cv)

        g = QGridLayout()
        g.addWidget(QLabel("Lens Type:"), 0, 0)
        self.cb_lens = QComboBox(); self.cb_lens.addItems(["Converging (f>0)", "Diverging (f<0)"])
        self.cb_lens.currentIndexChanged.connect(self._upd); g.addWidget(self.cb_lens, 0, 1)

        g.addWidget(QLabel("Focal Length (px):"), 1, 0)
        self.sl_f = make_slider(50, 250, 120, self._upd); g.addWidget(self.sl_f, 1, 1)
        self.lbl_f = QLabel("120"); g.addWidget(self.lbl_f, 1, 2)

        g.addWidget(QLabel("Object Distance (px):"), 2, 0)
        self.sl_do = make_slider(50, 400, 250, self._upd); g.addWidget(self.sl_do, 2, 1)
        self.lbl_do = QLabel("250"); g.addWidget(self.lbl_do, 2, 2)

        g.addWidget(QLabel("Object Height (px):"), 3, 0)
        self.sl_h = make_slider(20, 120, 60, self._upd); g.addWidget(self.sl_h, 3, 1)
        ly.addLayout(g)

        self.info = info_label(); ly.addWidget(self.info); self._upd(); ly.addStretch()

    def _upd(self):
        lt = self.cb_lens.currentIndex(); f_px = self.sl_f.value()
        do_px = self.sl_do.value(); h_px = self.sl_h.value()
        self.lbl_f.setText(str(f_px)); self.lbl_do.setText(str(do_px))
        self.cv.lens_type = lt; self.cv.focal_len = f_px
        self.cv.obj_dist = do_px; self.cv.obj_height = h_px; self.cv.update()

        f = f_px if lt == 0 else -f_px
        if abs(do_px - f) > 1:
            di = f * do_px / (do_px - f); m = -di / do_px
            img_h = m * h_px
            itype = "Real" if di > 0 else "Virtual"
            orient = "Inverted" if m < 0 else "Upright"
            size = "Magnified" if abs(m) > 1 else ("Same size" if abs(m) == 1 else "Diminished")
            self.info.setText(
                f"1/f = 1/dₒ + 1/dᵢ  →  1/{f} = 1/{do_px} + 1/dᵢ\n"
                f"dᵢ = {di:.1f} px  |  m = {m:.3f}\n"
                f"Image height = {img_h:.1f} px\n"
                f"Type: {itype} | {orient} | {size}")
        else:
            self.info.setText("Object at focal point → Image at infinity!")


class PhotonCalcPage(QWidget):
    """Photon energy/frequency/wavelength converter."""
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(12)
        ly.addWidget(heading("⚛️ Photon Calculator"))
        ly.addWidget(body(
            "E = hf = hc/λ.  Enter any one value and the others are computed automatically.\n"
            "h = 6.626×10⁻³⁴ J·s  |  c = 2.998×10⁸ m/s  |  1 eV = 1.602×10⁻¹⁹ J"))

        # Input section
        g = QGroupBox("Input (select type, then adjust slider)")
        gl = QGridLayout(g)

        self.cb_input = QComboBox()
        self.cb_input.addItems(["Wavelength (nm)", "Frequency (THz)", "Energy (eV)"])
        self.cb_input.currentIndexChanged.connect(self._mode_changed)
        gl.addWidget(self.cb_input, 0, 0, 1, 3)

        # Wavelength slider
        gl.addWidget(QLabel("Wavelength (nm):"), 1, 0)
        self.sl_wl = make_slider(1, 1500, 550, self._from_wl)
        gl.addWidget(self.sl_wl, 1, 1)

        # Frequency slider
        gl.addWidget(QLabel("Frequency (THz):"), 2, 0)
        self.sl_f = make_slider(1, 30000, 545, self._from_freq)
        gl.addWidget(self.sl_f, 2, 1)

        # Energy slider
        gl.addWidget(QLabel("Energy (eV):"), 3, 0)
        self.sl_e = make_slider(1, 5000, 225, self._from_energy)
        gl.addWidget(self.sl_e, 3, 1)

        ly.addWidget(g)

        # Color preview
        self.color_box = QLabel(); self.color_box.setFixedHeight(40)
        self.color_box.setStyleSheet("border-radius:8px;border:2px solid #444;"); ly.addWidget(self.color_box)

        # Spectrum bar
        self.bar = SpectrumBar(); self.bar.on_select = self._bar_select; ly.addWidget(self.bar)

        # Results
        self.info = info_label(); ly.addWidget(self.info)
        ly.addStretch()
        self._from_wl()

    def _mode_changed(self): pass

    def _bar_select(self, wl):
        self.sl_wl.setValue(int(wl)); self._from_wl()

    def _from_wl(self):
        wl = max(1, self.sl_wl.value())
        f = wl_to_freq(wl); e = wl_to_energy(wl)
        self.sl_f.blockSignals(True); self.sl_f.setValue(int(f/1e12)); self.sl_f.blockSignals(False)
        self.sl_e.blockSignals(True); self.sl_e.setValue(int(e*100)); self.sl_e.blockSignals(False)
        self._update_display(wl, f, e)

    def _from_freq(self):
        f_thz = max(1, self.sl_f.value())
        f = f_thz * 1e12; wl = C/f*1e9; e = H*f/EV
        self.sl_wl.blockSignals(True); self.sl_wl.setValue(int(wl)); self.sl_wl.blockSignals(False)
        self.sl_e.blockSignals(True); self.sl_e.setValue(int(e*100)); self.sl_e.blockSignals(False)
        self._update_display(wl, f, e)

    def _from_energy(self):
        e_ev = max(0.01, self.sl_e.value()/100.)
        wl = H*C/(e_ev*EV)*1e9; f = C/(wl*1e-9)
        self.sl_wl.blockSignals(True); self.sl_wl.setValue(int(wl)); self.sl_wl.blockSignals(False)
        self.sl_f.blockSignals(True); self.sl_f.setValue(int(f/1e12)); self.sl_f.blockSignals(False)
        self._update_display(wl, f, e_ev)

    def _update_display(self, wl_nm, f_hz, e_ev):
        cn = color_name(wl_nm) if 380<=wl_nm<=780 else em_region(wl_nm)
        if 380<=wl_nm<=780:
            rgb = wavelength_to_rgb(wl_nm)
            self.color_box.setStyleSheet(f"background-color:{rgb.name()};border-radius:8px;border:2px solid #444;")
        else:
            self.color_box.setStyleSheet("background-color:#222;border-radius:8px;border:2px solid #444;")
        self.bar.selected_wl = max(380, min(780, wl_nm)); self.bar.update()

        # Momentum
        p = H / (wl_nm * 1e-9)  # kg·m/s

        self.info.setText(
            f"🔬 Wavelength:  λ = {wl_nm:.1f} nm  =  {wl_nm*1e-9:.3e} m\n"
            f"📡 Frequency:   f = {f_hz:.3e} Hz  =  {f_hz/1e12:.1f} THz\n"
            f"⚡ Energy:      E = {e_ev:.4f} eV  =  {e_ev*EV:.3e} J\n"
            f"📐 Momentum:    p = h/λ = {p:.3e} kg·m/s\n"
            f"🎨 Color:       {cn}\n"
            f"📂 EM Region:   {em_region(wl_nm)}\n\n"
            f"Equations:\n"
            f"  E = hf = hc/λ\n"
            f"  c = λf = {C:.3e} m/s\n"
            f"  p = h/λ  (photon momentum)")


class RayleighPage(QWidget):
    """Rayleigh scattering — why the sky is blue."""
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(8)
        ly.addWidget(heading("🌤️ Rayleigh Scattering"))
        ly.addWidget(body(
            "<b>Rayleigh scattering:</b> I_scat ∝ 1/λ⁴ — shorter wavelengths scatter much more. "
            "This is why the sky is blue (scattered sunlight) and sunsets are red (long path "
            "removes blue, leaving red/orange). Particles must be ≪ λ (air molecules)."))

        self.cv = RayleighSkyWidget(); ly.addWidget(self.cv)

        g = QGridLayout()
        g.addWidget(QLabel("Sun Elevation (°):"), 0, 0)
        self.sl_elev = make_slider(2, 90, 45, self._upd)
        g.addWidget(self.sl_elev, 0, 1)

        g.addWidget(QLabel("Atmosphere Thickness:"), 1, 0)
        self.sl_path = make_slider(5, 30, 10, self._upd)
        g.addWidget(self.sl_path, 1, 1)
        ly.addLayout(g)

        # Scattering comparison
        self.scat_info = info_label(); ly.addWidget(self.scat_info)
        self._upd(); ly.addStretch()

    def _upd(self):
        elev = self.sl_elev.value(); path = self.sl_path.value() / 10.
        self.cv.sun_elevation = elev; self.cv.path_length = path; self.cv.update()

        # Compute relative scattering for key colors
        wls = np.array([400., 450., 500., 550., 600., 650., 700.], dtype=np.float64)
        scat = rayleigh_scattering_numba(wls)

        lines = ["Relative scattering (1/λ⁴, normalized to 550nm green):\n"]
        for wl, s in zip(wls, scat):
            cn = color_name(wl)
            bar = "█" * int(s * 5)
            lines.append(f"  {wl:.0f}nm {cn:7s}: {s:.3f}×  {bar}")

        path_mult = path / math.sin(math.radians(max(1, elev)))
        lines.append(f"\nAtm. path: {path_mult:.1f}× (elevation={elev}°)")
        lines.append(f"  → At horizon: long path, blue scattered away → red/orange sunset")
        lines.append(f"  → At zenith: short path, blue dominates → blue sky")

        self.scat_info.setText("\n".join(lines))


class FresnelPage(QWidget):
    """Fresnel equations — reflection/transmission coefficients."""
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(8)
        ly.addWidget(heading("📊 Fresnel Equations"))
        ly.addWidget(body(
            "<b>Fresnel equations</b> give the reflection and transmission coefficients at an interface:\n"
            "  Rs = |( n₁cosθᵢ − n₂cosθₜ )/( n₁cosθᵢ + n₂cosθₜ )|²   (s-polarization)\n"
            "  Rp = |( n₂cosθᵢ − n₁cosθₜ )/( n₂cosθᵢ + n₁cosθₜ )|²   (p-polarization)\n\n"
            "<b>Brewster's angle:</b> θ_B = arctan(n₂/n₁) — Rp = 0, perfect s-polarization.\n"
            "<b>Critical angle:</b> (when n₁>n₂) θ_c = arcsin(n₂/n₁) — total internal reflection."))

        self.cv = FresnelWidget(); ly.addWidget(self.cv)

        g = QGridLayout()
        g.addWidget(QLabel("n₁ (incident medium):"), 0, 0)
        self.cb1 = QComboBox(); self.cb1.addItems(MEDIA.keys())
        self.cb1.setCurrentText("Air"); self.cb1.currentTextChanged.connect(self._upd); g.addWidget(self.cb1, 0, 1)
        self.lbl_n1 = QLabel("1.0003"); g.addWidget(self.lbl_n1, 0, 2)

        g.addWidget(QLabel("n₂ (transmitted medium):"), 1, 0)
        self.cb2 = QComboBox(); self.cb2.addItems(MEDIA.keys())
        self.cb2.setCurrentText("Glass (Crown)"); self.cb2.currentTextChanged.connect(self._upd); g.addWidget(self.cb2, 1, 1)
        self.lbl_n2 = QLabel("1.5200"); g.addWidget(self.lbl_n2, 1, 2)
        ly.addLayout(g)

        self.info = info_label(); ly.addWidget(self.info); self._upd(); ly.addStretch()

    def _upd(self):
        n1 = MEDIA.get(self.cb1.currentText(), 1.); n2 = MEDIA.get(self.cb2.currentText(), 1.)
        self.lbl_n1.setText(f"{n1:.4f}"); self.lbl_n2.setText(f"{n2:.4f}")
        self.cv.n1 = n1; self.cv.n2 = n2; self.cv.update()

        brewster = math.degrees(math.atan(n2/n1))
        lines = [f"n₁ = {n1:.4f}  |  n₂ = {n2:.4f}"]
        lines.append(f"Brewster's angle: θ_B = arctan(n₂/n₁) = {brewster:.1f}°")

        if n1 > n2:
            critical = math.degrees(math.asin(n2/n1))
            lines.append(f"Critical angle: θ_c = arcsin(n₂/n₁) = {critical:.1f}°  →  TIR above this angle")
        else:
            lines.append("No critical angle (n₁ ≤ n₂, no TIR possible)")

        # Normal incidence reflectance
        R_normal = ((n1-n2)/(n1+n2))**2
        lines.append(f"Normal incidence reflectance: R = {R_normal:.4f} ({R_normal*100:.2f}%)")
        lines.append(f"Normal incidence transmittance: T = {1-R_normal:.4f} ({(1-R_normal)*100:.2f}%)")

        self.info.setText("\n".join(lines))


class QuizPage(QWidget):
    QS = [
        {"q":"Speed of light in vacuum?","o":["3×10⁸ m/s","3×10⁶ m/s","3×10¹⁰ m/s","3×10⁴ m/s"],"a":0,"e":"c ≈ 2.998×10⁸ m/s."},
        {"q":"Which visible color has longest wavelength?","o":["Violet","Green","Red","Blue"],"a":2,"e":"Red ~620-780 nm."},
        {"q":"Red + Green light (additive) = ?","o":["Blue","Yellow","Magenta","Cyan"],"a":1,"e":"R+G=Yellow in additive mixing."},
        {"q":"Snell's Law:","o":["n₁sinθ₁=n₂sinθ₂","n₁cosθ₁=n₂cosθ₂","n₁θ₁=n₂θ₂","sinθ₁/sinθ₂=n₂/n₁"],"a":0,"e":"n₁sinθ₁=n₂sinθ₂."},
        {"q":"Total internal reflection occurs when:","o":["Light hits a mirror","θ>critical, n₁>n₂","Light hits rough surface","n₁<n₂"],"a":1,"e":"TIR: denser→rarer, θ>critical."},
        {"q":"Highest photon energy:","o":["Radio","Visible","X-rays","Infrared"],"a":2,"e":"E=hf; X-rays highest f→highest E."},
        {"q":"Dispersion in a prism caused by:","o":["n varies with λ","Prism absorbs colors","Reflection","Frequency changes"],"a":0,"e":"n(λ) varies→different bending per color."},
        {"q":"Visible light range:","o":["380-780 nm","100-400 nm","780-1000 nm","10-380 nm"],"a":0,"e":"380-780 nm."},
        {"q":"Reflection angles measured from:","o":["Surface","Normal","Horizontal","Ray"],"a":1,"e":"From normal (perpendicular)."},
        {"q":"Highest frequency visible color:","o":["Red","Green","Yellow","Violet"],"a":3,"e":"Violet: shortest λ→highest f."},
        {"q":"Light entering glass from air:","o":["Speeds up","Slows down","Same speed","Stops"],"a":1,"e":"v=c/n; glass n≈1.52→slows."},
        {"q":"Single-slit diffraction:","o":["I₀sinc²(β)","I₀cos²(β)","I₀sin²(β)","I₀β²"],"a":0,"e":"I=I₀[sin(β)/β]²."},
        {"q":"Wien's displacement law:","o":["λ_max=b/T","λ_max=bT","λ_max=T/b","λ_max=b/T²"],"a":0,"e":"λ_max=2.898×10⁻³/T."},
        {"q":"Malus's Law:","o":["I=I₀cos²θ","I=I₀sin²θ","I=I₀cosθ","I=I₀/sinθ"],"a":0,"e":"I=I₀cos²θ for polarized light through polarizer."},
        {"q":"Brewster's angle:","o":["arctan(n₂/n₁)","arcsin(n₂/n₁)","arccos(n₂/n₁)","arctan(n₁/n₂)"],"a":0,"e":"θ_B=arctan(n₂/n₁) where Rp=0."},
        {"q":"Rayleigh scattering intensity:","o":["∝ 1/λ⁴","∝ 1/λ²","∝ λ⁴","∝ λ²"],"a":0,"e":"I_scat ∝ 1/λ⁴ → blue scatters more."},
        {"q":"Thin lens equation:","o":["1/f=1/dₒ+1/dᵢ","f=dₒ+dᵢ","1/f=dₒ×dᵢ","f=dₒ/dᵢ"],"a":0,"e":"1/f=1/dₒ+1/dᵢ."},
        {"q":"Stefan-Boltzmann law:","o":["P=σT⁴","P=σT²","P=σ/T⁴","P=σT"],"a":0,"e":"Total radiated power P=σT⁴."},
    ]
    def __init__(self):
        super().__init__()
        ly=QVBoxLayout(self); ly.setContentsMargins(20,20,20,20); ly.setSpacing(12)
        ly.addWidget(heading("📝 Quiz — Test Your Knowledge"))
        self.score_l=QLabel("Score: 0/0"); self.score_l.setStyleSheet("font-size:14px;color:#80d0ff;font-weight:bold;"); ly.addWidget(self.score_l)
        self.q_l=QLabel(); self.q_l.setWordWrap(True); self.q_l.setStyleSheet("font-size:15px;color:#e0e0f0;padding:10px;"); ly.addWidget(self.q_l)
        self.bg=QButtonGroup(self); self.rbs=[]
        for i in range(4):
            r=QRadioButton(); r.setStyleSheet("font-size:13px;color:#c0c0d8;padding:4px;")
            self.bg.addButton(r,i); self.rbs.append(r); ly.addWidget(r)
        self.sb=QPushButton("Submit"); self.sb.clicked.connect(self._submit)
        self.sb.setStyleSheet("background:#0f4460;color:white;border-radius:6px;padding:10px 24px;font-weight:bold;"); ly.addWidget(self.sb)
        self.fb=info_label(); ly.addWidget(self.fb)
        self.nb=QPushButton("Next →"); self.nb.clicked.connect(self._next)
        self.nb.setStyleSheet("background:#1a6040;color:white;border-radius:6px;padding:10px 24px;font-weight:bold;")
        self.nb.setVisible(False); ly.addWidget(self.nb)
        self.sc=0; self.tot=0; self.cur=0
        self.order=list(range(len(self.QS))); random.shuffle(self.order); self._show(); ly.addStretch()

    def _show(self):
        q=self.QS[self.order[self.cur%len(self.order)]]
        self.q_l.setText(f"Q{self.tot+1}: {q['q']}")
        for i,r in enumerate(self.rbs): r.setText(q['o'][i]); r.setChecked(False); r.setEnabled(True)
        self.fb.setText(""); self.sb.setVisible(True); self.nb.setVisible(False)

    def _submit(self):
        c=self.bg.checkedId()
        if c<0: self.fb.setText("⚠️ Select an answer."); return
        q=self.QS[self.order[self.cur%len(self.order)]]; ok=c==q['a']; self.tot+=1
        if ok: self.sc+=1; self.fb.setText(f"✅ Correct!\n\n{q['e']}"); self.fb.setStyleSheet("background:#0a2a1a;border:1px solid #1a5a3a;border-radius:6px;padding:10px;font-size:13px;color:#80e0a0;")
        else: self.fb.setText(f"❌ Answer: {q['o'][q['a']]}\n\n{q['e']}"); self.fb.setStyleSheet("background:#2a0a0a;border:1px solid #5a1a1a;border-radius:6px;padding:10px;font-size:13px;color:#e08080;")
        self.score_l.setText(f"Score: {self.sc}/{self.tot}")
        for r in self.rbs: r.setEnabled(False)
        self.sb.setVisible(False); self.nb.setVisible(True)

    def _next(self):
        self.cur+=1
        if self.cur>=len(self.order): random.shuffle(self.order); self.cur=0
        self._show()


# ═══════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════

DARK_STYLE="""
QMainWindow{background:#10101e}
QWidget{background:#10101e;color:#d0d0e0;font-family:'Segoe UI',Arial}
QListWidget{background:#14142a;border:none;border-radius:8px;padding:6px;outline:none}
QListWidget::item{padding:11px 12px;border-radius:6px;margin:1px 0;color:#8888a8;font-size:12px}
QListWidget::item:selected{background:#1a3060;color:#fff}
QListWidget::item:hover{background:#1a1a40}
QSlider::groove:horizontal{height:6px;background:#2a2a4a;border-radius:3px}
QSlider::handle:horizontal{background:#4a90d9;width:16px;height:16px;margin:-5px 0;border-radius:8px}
QSlider::sub-page:horizontal{background:#3a6ab0;border-radius:3px}
QComboBox{background:#1a1a3a;border:1px solid #2a2a5a;border-radius:4px;padding:5px 10px;color:#d0d0e0}
QComboBox::drop-down{border:none}
QComboBox QAbstractItemView{background:#1a1a3a;color:#d0d0e0;selection-background-color:#2a4a7a}
QLabel{background:transparent}
QGroupBox{border:1px solid #2a2a5a;border-radius:8px;margin-top:14px;padding-top:18px;color:#9090b0}
QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 6px}
QRadioButton::indicator{width:16px;height:16px}
QPushButton{background:#1a3060;color:white;border:none;border-radius:5px;padding:8px 16px;font-weight:bold}
QPushButton:hover{background:#2a4080}
QScrollBar:vertical{background:#14142a;width:8px;border:none}
QScrollBar::handle:vertical{background:#2a2a5a;border-radius:4px;min-height:30px}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("✨ Light & Color — GPU-Accelerated Learning App")
        self.setMinimumSize(1150,780); self.resize(1400,900)
        self.setStyleSheet(DARK_STYLE)

        central=QWidget(); self.setCentralWidget(central)
        hl=QHBoxLayout(central); hl.setContentsMargins(8,8,8,8); hl.setSpacing(8)

        self.sidebar=QListWidget(); self.sidebar.setFixedWidth(210)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        items=["🌈  EM Spectrum","🎨  Color Mixing","🌊  Wave Properties",
               "🪞  Reflection","🔀  Refraction","🔬  Dispersion",
               "⚡  Diffraction [GPU]","🌊  Wave Sim [GPU]",
               "🌡️  Blackbody Rad.","↕️  Polarization",
               "🔭  Lens Simulator","⚛️  Photon Calculator",
               "🌤️  Rayleigh Scatter","📊  Fresnel Equations",
               "📝  Quiz"]
        for i in items: self.sidebar.addItem(i)
        self.sidebar.setCurrentRow(0)
        self.sidebar.currentRowChanged.connect(self._sw)
        hl.addWidget(self.sidebar)

        self.stack=QStackedWidget()
        self.pages=[SpectrumPage(),ColorMixingPage(),WavePage(),
                    ReflectionPage(),RefractionPage(),DispersionPage(),
                    DiffractionPage(),WaveSimPage(),
                    BlackbodyPage(),PolarizationPage(),
                    LensPage(),PhotonCalcPage(),
                    RayleighPage(),FresnelPage(),
                    QuizPage()]
        for p in self.pages: self.stack.addWidget(p)
        hl.addWidget(self.stack,1)

        be="🟢 CuPy GPU Acceleration" if HAS_CUPY else "🔵 NumPy CPU (install cupy-cuda12x for GPU)"
        sb=self.statusBar(); sb.showMessage(f"Backend: {be}  |  Numba JIT: ✓  |  15 interactive modules")
        sb.setStyleSheet("color:#7090b0;font-size:11px;")

    def _sw(self, i): self.stack.setCurrentIndex(i)


if __name__=="__main__":
    app=QApplication(sys.argv); app.setStyle("Fusion")
    win=MainWindow(); win.show(); sys.exit(app.exec())