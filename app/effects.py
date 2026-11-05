"""Weather and air-quality visual effects for the dashboard (pure HTML + CSS)."""
import random
from datetime import timedelta

import pandas as pd

CSS = """
<style>
.scene{position:relative;min-height:250px;border-radius:18px;overflow:hidden;margin-bottom:14px;color:#fff;box-shadow:0 6px 24px rgba(0,0,0,.18);animation:sceneIn .8s ease both}
@keyframes sceneIn{from{opacity:0}to{opacity:1}}
.fxlayer{position:absolute;inset:0;overflow:hidden}
.scene-text{position:relative;z-index:5;padding:20px 26px;text-shadow:0 2px 10px rgba(0,0,0,.45)}
.scene-top{font-size:1rem;opacity:.95}
.scene-time{position:absolute;right:18px;bottom:14px;z-index:5;font-size:.85rem;background:rgba(0,0,0,.30);padding:5px 13px;border-radius:999px;text-shadow:none}
@media (max-width:760px){.scene-time{position:relative;display:inline-block;margin:0 0 14px 26px}}
.scene-big{font-size:3.4rem;font-weight:800;line-height:1.05}
.scene-big span{font-size:1.05rem;font-weight:500;margin-left:6px}
.pill{display:inline-block;padding:4px 14px;border-radius:999px;font-weight:700;margin:6px 0;text-shadow:none}
.pill.pulse2{animation:pillpulse 2.2s infinite}
@keyframes pillpulse{0%{box-shadow:0 0 0 0 rgba(255,255,255,.6)}70%{box-shadow:0 0 0 14px rgba(255,255,255,0)}100%{box-shadow:0 0 0 0 rgba(255,255,255,0)}}
.chip{display:inline-block;background:rgba(255,255,255,.2);border-radius:999px;padding:3px 11px;margin:6px 6px 0 0;font-size:.85rem;text-shadow:none}
.plainhero{border-radius:18px;padding:22px 28px;margin-bottom:14px}
.plainhero .big{font-size:3.4rem;font-weight:800;line-height:1.1}
.sun{position:absolute;top:24px;right:12%;width:86px;height:86px;border-radius:50%;background:radial-gradient(circle,#fff7b8 0%,#ffd84a 55%,#ffb300 100%);box-shadow:0 0 60px 26px rgba(255,200,50,.55);animation:sunpulse 5s ease-in-out infinite}
.sun:before{content:"";position:absolute;inset:-70px;border-radius:50%;background:repeating-conic-gradient(rgba(255,235,120,.40) 0 5deg,transparent 5deg 15deg);-webkit-mask-image:radial-gradient(circle,transparent 30%,#000 38%,transparent 70%);mask-image:radial-gradient(circle,transparent 30%,#000 38%,transparent 70%);animation:spin 50s linear infinite}
.sun.lv3{background:radial-gradient(circle,#ffe2a0,#ffb040 60%,#ff8a1f);box-shadow:0 0 50px 20px rgba(255,150,50,.4)}
.sun.lv4,.sun.lv5{background:radial-gradient(circle,#ffc9a8,#f0703a 60%,#c94a22);box-shadow:0 0 40px 14px rgba(220,90,40,.35)}
.sun.lv3:before,.sun.lv4:before,.sun.lv5:before{opacity:.3}
@keyframes sunpulse{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
@keyframes spin{to{transform:rotate(360deg)}}
.moon{position:absolute;top:26px;right:13%;width:64px;height:64px;border-radius:50%;box-shadow:inset -16px -4px 0 0 #f4f1de;filter:drop-shadow(0 0 14px rgba(244,241,222,.55));animation:sunpulse 6s ease-in-out infinite}
.star{position:absolute;width:var(--sz);height:var(--sz);background:#fff;border-radius:50%;animation:twinkle 3s ease-in-out var(--dl) infinite}
@keyframes twinkle{0%,100%{opacity:.2}50%{opacity:1}}
.cloud{position:absolute;left:-320px;top:var(--t);width:var(--w);height:calc(var(--w) * .28);background:var(--c);opacity:var(--o);border-radius:999px;box-shadow:0 8px 14px rgba(0,0,0,.10);animation:drift var(--dur) linear var(--dl) infinite}
.cloud:before,.cloud:after{content:"";position:absolute;background:inherit;border-radius:50%}
.cloud:before{width:calc(var(--w) * .42);height:calc(var(--w) * .42);top:calc(var(--w) * -.18);left:14%}
.cloud:after{width:calc(var(--w) * .32);height:calc(var(--w) * .32);top:calc(var(--w) * -.12);right:16%}
@keyframes drift{from{transform:translateX(0)}to{transform:translateX(calc(100vw + 700px))}}
.drop{position:absolute;top:-40px;left:var(--x);width:1.6px;height:var(--len);opacity:var(--o);background:linear-gradient(to bottom,rgba(200,220,245,0),rgba(200,220,245,.9));animation:fall var(--dur) linear var(--dl) infinite}
@keyframes fall{from{transform:translate(0,0) rotate(var(--r))}to{transform:translate(var(--dx),var(--h)) rotate(var(--r))}}
.scene .drop{--h:300px}
.fx-page .drop{--h:115vh}
.flake{position:absolute;top:-10px;left:var(--x);width:var(--sz);height:var(--sz);background:#fff;border-radius:50%;opacity:.85;animation:snowfall var(--dur) linear var(--dl) infinite}
@keyframes snowfall{0%{transform:translate(0,0)}50%{transform:translate(var(--sw),calc(var(--h) * .5))}100%{transform:translate(0,var(--h))}}
.scene .flake{--h:270px}
.fx-page .flake{--h:110vh}
.fogband{position:absolute;left:-50%;width:200%;top:var(--t);height:70px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.55),rgba(255,255,255,.25),rgba(255,255,255,.6),transparent);filter:blur(14px);animation:fogmove var(--dur) ease-in-out var(--dl) infinite alternate}
@keyframes fogmove{from{transform:translateX(-8%)}to{transform:translateX(8%)}}
.flash{position:absolute;inset:0;background:#fff;opacity:0;animation:flash 6.5s linear var(--dl) infinite}
@keyframes flash{0%,90%,100%{opacity:0}91%{opacity:.7}92%{opacity:.05}93.5%{opacity:.5}95%{opacity:0}}
.fx-page .flash{animation-name:flashpage}
@keyframes flashpage{0%,90%,100%{opacity:0}91%{opacity:.16}92%{opacity:.02}93.5%{opacity:.12}95%{opacity:0}}
.smog{position:absolute;inset:0}
.blob{position:absolute;left:-30%;top:var(--t);width:var(--s);height:var(--s);border-radius:50%;background:radial-gradient(circle,var(--c),transparent 68%);filter:blur(12px);animation:blobdrift var(--dur) linear var(--dl) infinite}
@keyframes blobdrift{from{transform:translateX(0)}to{transform:translateX(calc(100vw + 40%))}}
.dust{position:absolute;left:var(--x);top:var(--y);width:var(--sz);height:var(--sz);border-radius:50%;background:rgba(120,95,70,.75);animation:dustfloat var(--dur) ease-in-out var(--dl) infinite}
@keyframes dustfloat{0%{transform:translate(0,0);opacity:0}20%{opacity:.85}100%{transform:translate(var(--dx),var(--dy));opacity:0}}
.mote{position:absolute;left:var(--x);bottom:-10px;width:var(--sz);height:var(--sz);border-radius:50%;background:rgba(255,255,255,.9);box-shadow:0 0 8px 2px rgba(255,255,255,.7);animation:moteup var(--dur) ease-in var(--dl) infinite}
@keyframes moteup{0%{transform:translate(0,0);opacity:0}15%{opacity:.9}100%{transform:translate(var(--dx),-260px);opacity:0}}
.fx-page{position:fixed;inset:0;pointer-events:none;z-index:40;overflow:hidden}
.fx-page .pagehaze{position:absolute;inset:0}
.fx-page .vignette{position:absolute;inset:0;background:radial-gradient(ellipse at center,transparent 55%,rgba(130,50,40,.22) 100%);animation:breathe 6s ease-in-out infinite}
@keyframes breathe{0%,100%{opacity:.55}50%{opacity:1}}
@media (prefers-reduced-motion:reduce){.fx-page *,.scene *{animation:none!important}.fx-page .drop,.fx-page .flake{display:none}}
</style>
"""

# sky gradients: kind -> (day (top, bottom), night (top, bottom))
SKY = {
    "clear": (("#3b97f5", "#a9d8ff"), ("#070b24", "#2a3470")),
    "partly": (("#4d8fd1", "#b5d3ee"), ("#0e1430", "#2f3a63")),
    "overcast": (("#76838f", "#b4bec7"), ("#1a2030", "#394157")),
    "rain": (("#46576a", "#7b8ea0"), ("#141a26", "#2e394a")),
    "storm": (("#262d3a", "#4b5666"), ("#0b0e16", "#262d3d")),
    "snow": (("#8199ae", "#dbe5ee"), ("#1b2535", "#46566c")),
    "fog": (("#a5aeb7", "#d9dde0"), ("#2a3038", "#58606a")),
}
# kind -> (cloud count, day colour, night colour, opacity)
CLOUDS = {
    "clear": (0, "#ffffff", "#6b7396", .9),
    "partly": (3, "#ffffff", "#6b7396", .92),
    "overcast": (6, "#d5dbe0", "#4a526b", .95),
    "rain": (6, "#9aa6b2", "#3a4254", .95),
    "storm": (7, "#59626f", "#2a303c", .97),
    "snow": (5, "#e6edf3", "#56627a", .92),
    "fog": (2, "#e9edf0", "#59626f", .55),
}
SMOG_RGB = [(0, 0, 0), (205, 185, 120), (200, 160, 90), (165, 115, 75), (125, 80, 105), (95, 45, 45)]
SCENE_HAZE = [0, .14, .26, .40, .52, .62]
PAGE_HAZE = [0, .02, .04, .07, .10, .14]
BLOBS = [0, 3, 6, 9, 12, 16]
DUST = [0, 0, 0, 16, 26, 38]


def level_of(pm):
    for i, upper in enumerate([9.0, 35.4, 55.4, 125.4, 225.4]):
        if pm <= upper:
            return i
    return 5


def is_daytime(time, lon):
    local = (int(getattr(time, "hour", 12)) + round(lon / 15)) % 24
    return 6 <= local < 18


def classify(wx, day):
    """Turn weather numbers into one scene type: (kind, label, emoji)."""
    if wx is None:
        return "partly", "Weather unavailable", "⛅"
    prec, cloud = float(wx["precipitation"]), float(wx["cloud_cover"])
    hum, temp, wind = float(wx["relative_humidity_2m"]), float(wx["temperature_2m"]), float(wx["wind_speed_10m"])
    if prec >= 0.1 and temp <= 0.5:
        return "snow", "Snow", "❄️"
    if prec >= 4 and cloud >= 80:
        return "storm", "Thunderstorm", "⛈️"
    if prec >= 0.1:
        return "rain", ("Light rain" if prec < 1 else "Rain" if prec < 4 else "Heavy rain"), "🌧️"
    if hum >= 96 and wind < 8:
        return "fog", "Fog", "🌫️"
    if cloud >= 75:
        return "overcast", "Overcast", "☁️"
    if cloud >= 35:
        return "partly", "Partly cloudy", "⛅" if day else "☁️"
    return "clear", ("Sunny" if day else "Clear night"), ("☀️" if day else "🌙")


def _rgba(rgb, a):
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"


def _precip_layers(rng, wx, kind, page):
    """Rain drops or snow flakes. page=True makes the lighter page-wide version."""
    prec = float(wx["precipitation"]) if wx is not None else 0.0
    wind = float(wx["wind_speed_10m"]) if wx is not None else 0.0
    n = 35 if prec < 1 else 65 if prec < 4 else 100
    n = int(n * (0.6 if page else 1.0))
    out = []
    if kind == "snow":
        for _ in range(int(n * 0.7)):
            out.append(f'<span class="flake" style="--x:{rng.uniform(0, 100):.1f}%;--sz:{rng.uniform(2.5, 6):.1f}px;--dur:{rng.uniform(5, 10):.1f}s;--dl:-{rng.uniform(0, 10):.1f}s;--sw:{rng.uniform(-30, 30):.0f}px"></span>')
        return out
    rot = min(wind * 0.5, 22)
    heavy = prec >= 4
    for _ in range(n):
        dur = rng.uniform(0.5, 0.8) if heavy else rng.uniform(0.8, 1.3)
        o = rng.uniform(0.15, 0.4) if page else rng.uniform(0.35, 0.85)
        dx = f"{-rot * 1.4:.0f}vh" if page else f"{-rot * 6:.0f}px"
        out.append(f'<span class="drop" style="--x:{rng.uniform(0, 100):.1f}%;--len:{rng.uniform(12, 24):.0f}px;--dur:{dur:.2f}s;--dl:-{rng.uniform(0, 1.5):.2f}s;--dx:{dx};--r:{rot:.0f}deg;--o:{o:.2f}"></span>')
    return out


def _smog_layers(rng, level, page):
    out = []
    rgb = SMOG_RGB[level]
    if level == 0:
        return out
    if page:
        out.append(f'<div class="pagehaze" style="background:{_rgba(rgb, 1)};opacity:{PAGE_HAZE[level]}"></div>')
    else:
        out.append(f'<div class="smog" style="background:linear-gradient(180deg,{_rgba(rgb, .35)},{_rgba(rgb, .95)});opacity:{SCENE_HAZE[level]}"></div>')
    for _ in range(BLOBS[level] if not page else max(2, BLOBS[level] // 2)):
        t = f"{rng.uniform(0, 80):.0f}vh" if page else f"{rng.uniform(5, 75):.0f}%"
        out.append(f'<span class="blob" style="--t:{t};--s:{rng.uniform(140, 320):.0f}px;--c:{_rgba(rgb, rng.uniform(.35, .6))};--dur:{rng.uniform(40, 90):.0f}s;--dl:-{rng.uniform(0, 90):.0f}s"></span>')
    for _ in range(DUST[level] if not page else DUST[level] // 2):
        out.append(f'<span class="dust" style="--x:{rng.uniform(0, 100):.1f}%;--y:{rng.uniform(0, 100):.1f}%;--sz:{rng.uniform(1.5, 3.5):.1f}px;--dur:{rng.uniform(6, 14):.1f}s;--dl:-{rng.uniform(0, 14):.1f}s;--dx:{rng.uniform(40, 160):.0f}px;--dy:{rng.uniform(-40, 40):.0f}px"></span>')
    if page and level >= 4:
        out.append('<div class="vignette"></div>')
    return out


def time_badge(data_time, lon):
    """'🕒 Thu 05 Nov, 14:00 local · 08:00 UTC · 3 h ago' (local time estimated from longitude)."""
    if data_time is None:
        return ""
    t = pd.Timestamp(data_time)
    local = t + timedelta(hours=int(round(lon / 15)))
    age = (pd.Timestamp.now("UTC").tz_localize(None) - t).total_seconds() / 3600
    ago = f"{age:.0f} h ago" if age < 48 else f"{age / 24:.0f} days ago"
    return (f'<div class="scene-time">🕒 {local:%a %d %b, %H:%M} local (approx.) · '
            f'{t:%H:%M} UTC · {ago}</div>')


def render_scene(city, pm, label, color, txt, wx, lon, mode, who=5.0, data_time=None):
    """Return one HTML string: CSS + (optional) page-wide effects + the weather scene."""
    level = level_of(pm)
    ratio = f"{pm / who:.1f}× the WHO annual guideline"
    if mode == "Off":
        html = (CSS + f'<div class="plainhero" style="background:linear-gradient(135deg,{color},{color}cc);color:{txt}">'
                f'<div style="opacity:.9">{city} · latest modeled PM2.5</div>'
                f'<div class="big">{pm:.0f} <span style="font-size:1.2rem;font-weight:500">µg/m³</span></div>'
                f'<div style="font-size:1.15rem">{label} · {ratio}</div>'
                + (f'<div style="margin-top:6px;font-size:.9rem;opacity:.9">🕒 {pd.Timestamp(data_time):%d %b %Y, %H:%M} UTC</div>' if data_time is not None else '')
                + '</div>')
        return " ".join(html.split())

    day = is_daytime(wx["time"], lon) if wx is not None else True
    kind, wlabel, emoji = classify(wx, day)
    rng = random.Random(f"{city}-{kind}-{level}-{int(day)}")
    wind = float(wx["wind_speed_10m"]) if wx is not None else 5.0
    sky = SKY[kind][0 if day else 1]
    parts = []

    if not day and kind in ("clear", "partly"):
        for _ in range(45):
            parts.append(f'<span class="star" style="left:{rng.uniform(0, 100):.1f}%;top:{rng.uniform(0, 75):.1f}%;--dl:-{rng.uniform(0, 4):.1f}s;--sz:{rng.choice([1, 1.5, 2])}px"></span>')
    if kind in ("clear", "partly"):
        parts.append(f'<div class="sun lv{level}"></div>' if day else '<div class="moon"></div>')

    n_cl, c_day, c_night, op = CLOUDS[kind]
    for _ in range(n_cl):
        dur = max(30, 110 - wind * 2.5) * rng.uniform(0.8, 1.3)
        parts.append(f'<span class="cloud" style="--t:{rng.uniform(45, 150):.0f}px;--w:{rng.uniform(140, 270):.0f}px;--c:{c_day if day else c_night};--o:{op};--dur:{dur:.0f}s;--dl:-{rng.uniform(0, dur):.0f}s"></span>')
    if kind == "fog":
        for i in range(3):
            parts.append(f'<span class="fogband" style="--t:{40 + i * 55}px;--dur:{rng.uniform(14, 24):.0f}s;--dl:-{rng.uniform(0, 14):.0f}s"></span>')
    if kind in ("rain", "storm", "snow"):
        parts += _precip_layers(rng, wx, kind, page=False)
    if kind == "storm":
        parts.append(f'<div class="flash" style="--dl:-{rng.uniform(0, 6):.1f}s"></div>')
    if level == 0 and day and kind in ("clear", "partly"):
        for _ in range(12):
            parts.append(f'<span class="mote" style="--x:{rng.uniform(0, 100):.1f}%;--sz:{rng.uniform(2, 4.5):.1f}px;--dur:{rng.uniform(6, 12):.1f}s;--dl:-{rng.uniform(0, 12):.1f}s;--dx:{rng.uniform(-40, 40):.0f}px"></span>')
    parts += _smog_layers(rng, level, page=False)

    page = []
    if mode == "Full":
        if kind in ("rain", "storm", "snow"):
            page += _precip_layers(rng, wx, kind, page=True)
        if kind == "storm":
            page.append(f'<div class="flash" style="--dl:-{rng.uniform(0, 6):.1f}s"></div>')
        page += _smog_layers(rng, level, page=True)

    chips = [f'<span class="chip">{emoji} {wlabel}</span>']
    if wx is not None:
        chips += [f'<span class="chip">🌡️ {float(wx["temperature_2m"]):.0f}°C</span>',
                  f'<span class="chip">💧 {float(wx["relative_humidity_2m"]):.0f}%</span>',
                  f'<span class="chip">💨 {wind:.0f} km/h</span>',
                  f'<span class="chip">☁️ {float(wx["cloud_cover"]):.0f}%</span>']
    pulse = " pulse2" if pm > 55.4 else ""
    html = (CSS
            + (f'<div class="fx-page">{"".join(page)}</div>' if page else "")
            + f'<div class="scene" style="background:linear-gradient(180deg,{sky[0]},{sky[1]})">'
              f'<div class="fxlayer">{"".join(parts)}</div>'
              f'<div class="scene-text"><div class="scene-top">{city} · latest modeled PM2.5</div>'
              f'<div class="scene-big">{pm:.0f}<span>µg/m³</span></div>'
              f'<span class="pill{pulse}" style="background:{color};color:{txt}">{label}</span> '
              f'<span style="font-size:.95rem">{ratio}</span>'
              f'<div>{"".join(chips)}</div></div>{time_badge(data_time, lon)}</div>')
    return " ".join(html.split())