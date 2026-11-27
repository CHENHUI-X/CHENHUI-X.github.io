import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

export interface GateMeta {
	postCount: number;
	topicCount: number;
	tagCount: number;
	lastUpdated: string;
	siteTitle: string;
}

interface Vec2 {
	x: number;
	y: number;
}

interface DriftSpec {
	bx: number;
	by: number;
	ax: number;
	ay: number;
	fx: number;
	fy: number;
	px: number;
	py: number;
	lockX?: boolean;
	lockY?: boolean;
}

interface ShardMorph {
	k: number;
	centroidWeights: Float32Array;
	bodyWeights: Float32Array;
	bodySource: Uint32Array;
	faceWeights: Float32Array;
	shadowWeights: Float32Array;
}

interface ShardPiece {
	id: string;
	isNav: boolean;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	group: any;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	bodyMesh: any;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	faceMesh: any;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	shadowMesh: any;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	bodyMat: any;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	faceMat: any;
	morph: ShardMorph | null;
	baseX: number;
	baseY: number;
	lift: number;
	liftVel: number;
	tipX: number;
	tipY: number;
	tipXVel: number;
	tipYVel: number;
	targetTipX: number;
	targetTipY: number;
	hoverBlend: number;
	flyDelay: number;
	flyDirX: number;
	flyDirY: number;
	flyDirZ: number;
	// biome-ignore lint/suspicious/noExplicitAny: three.js Vector3
	tumbleAxis: any;
	tumbleAngle: number;
}

const STORAGE_KEY = "chenhui_gate_seen";

const SHARD_TELEMETRY = [
	"CHENHUI · 格物 · 技术笔记",
	"01 / 数学基础 · 概率、信息论与线性代数",
	"02 / 机器学习 · 统计推断与优化方法",
	"03 / 深度学习 · 网络结构与训练原理",
	"04 / 大语言模型 · 注意力、显存与对齐",
	"点击棱镜进入博客",
];

const SCRAMBLE_GLYPHS = "0123456789ABCDEF∇∑∫∂λθΩπ/·";

function clamp(v: number, lo: number, hi: number): number {
	return Math.min(hi, Math.max(lo, v));
}

function easeOutBack(t: number): number {
	const c1 = 1.35;
	const c3 = c1 + 1;
	const x = t - 1;
	return 1 + c3 * x * x * x + c1 * x * x;
}

function easeInCubic(t: number): number {
	return t * t * t;
}

function hslToHex(h: number, s: number, l: number): string {
	const sn = s / 100;
	const ln = l / 100;
	const c = (1 - Math.abs(2 * ln - 1)) * sn;
	const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
	const m = ln - c / 2;
	let r = 0;
	let g = 0;
	let b = 0;
	if (h < 60) {
		r = c;
		g = x;
	} else if (h < 120) {
		r = x;
		g = c;
	} else if (h < 180) {
		g = c;
		b = x;
	} else if (h < 240) {
		g = x;
		b = c;
	} else if (h < 300) {
		r = x;
		b = c;
	} else {
		r = c;
		b = x;
	}
	const toHex = (n: number) =>
		Math.round((n + m) * 255)
			.toString(16)
			.padStart(2, "0");
	return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

/** Lightweight zero-asset Web Audio API crystal glass harmonic synthesizer (opt-in). */
class CrystalAudioSynth {
	private ctx: AudioContext | null = null;
	private enabled = false;
	private lastHoverMs = 0;
	// Pentatonic crystalline frequencies (C5, D5, E5, G5, A5, C6)
	private readonly scale = [523.25, 587.33, 659.25, 783.99, 880.0, 1046.5];

	isEnabled(): boolean {
		return this.enabled;
	}

	toggle(): boolean {
		this.enabled = !this.enabled;
		if (this.enabled) {
			this.ensureCtx();
			this.playHoverChime(5);
		}
		return this.enabled;
	}

	private ensureCtx(): AudioContext | null {
		if (!this.ctx) {
			const Ctor =
				window.AudioContext ||
				(window as unknown as { webkitAudioContext?: typeof AudioContext })
					.webkitAudioContext;
			if (!Ctor) return null;
			this.ctx = new Ctor();
		}
		if (this.ctx.state === "suspended") {
			void this.ctx.resume();
		}
		return this.ctx;
	}

	playHoverChime(index: number) {
		if (!this.enabled) return;
		const nowMs = performance.now();
		if (nowMs - this.lastHoverMs < 95) return;
		this.lastHoverMs = nowMs;
		const ctx = this.ensureCtx();
		if (!ctx) return;

		const t = ctx.currentTime;
		const freq = this.scale[Math.abs(index) % this.scale.length];

		const osc = ctx.createOscillator();
		const overtone = ctx.createOscillator();
		const gain = ctx.createGain();
		const overGain = ctx.createGain();

		osc.type = "sine";
		osc.frequency.setValueAtTime(freq, t);
		overtone.type = "triangle";
		overtone.frequency.setValueAtTime(freq * 2.756, t); // Glass bell inharmonic partial

		gain.gain.setValueAtTime(0.0001, t);
		gain.gain.exponentialRampToValueAtTime(0.045, t + 0.012);
		gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.55);

		overGain.gain.setValueAtTime(0.0001, t);
		overGain.gain.exponentialRampToValueAtTime(0.012, t + 0.008);
		overGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);

		osc.connect(gain);
		overtone.connect(overGain);
		overGain.connect(gain);
		gain.connect(ctx.destination);

		osc.start(t);
		overtone.start(t);
		osc.stop(t + 0.58);
		overtone.stop(t + 0.26);
	}

	playUnsealChord() {
		if (!this.enabled) return;
		const ctx = this.ensureCtx();
		if (!ctx) return;
		const t = ctx.currentTime;
		// Shimmering Cmaj9 crystal harp arpeggio
		const notes = [523.25, 659.25, 783.99, 987.77, 1174.66];
		for (let i = 0; i < notes.length; i++) {
			const start = t + i * 0.036;
			const osc = ctx.createOscillator();
			const gain = ctx.createGain();
			osc.type = "sine";
			osc.frequency.setValueAtTime(notes[i], start);
			osc.frequency.exponentialRampToValueAtTime(notes[i] * 1.006, start + 0.7);
			gain.gain.setValueAtTime(0.0001, start);
			gain.gain.exponentialRampToValueAtTime(0.048, start + 0.02);
			gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.95);
			osc.connect(gain);
			gain.connect(ctx.destination);
			osc.start(start);
			osc.stop(start + 1.0);
		}
	}

	dispose() {
		if (this.ctx && this.ctx.state !== "closed") {
			void this.ctx.close();
		}
		this.ctx = null;
	}
}

/** Ensure polygon is strictly counter-clockwise in 3D (x right, y up) space. */
function ensureCCW(pts: Vec2[]): Vec2[] {
	let area = 0;
	for (let i = 0; i < pts.length; i++) {
		const a = pts[i];
		const b = pts[(i + 1) % pts.length];
		area += a.x * b.y - b.x * a.y;
	}
	return area < 0 ? [...pts].reverse() : pts;
}

/** Shrink a CCW polygon inward along edge normals by `inset` distance. */
function insetPolygon(pts: Vec2[], inset: number): Vec2[] {
	const ccw = ensureCCW(pts);
	const n = ccw.length;
	const lines: Array<{ px: number; py: number; dx: number; dy: number }> = [];

	for (let i = 0; i < n; i++) {
		const a = ccw[i];
		const b = ccw[(i + 1) % n];
		const ex = b.x - a.x;
		const ey = b.y - a.y;
		const len = Math.hypot(ex, ey) || 1;
		const nx = -ey / len;
		const ny = ex / len;
		lines.push({
			px: a.x + nx * inset,
			py: a.y + ny * inset,
			dx: ex / len,
			dy: ey / len,
		});
	}

	const out: Vec2[] = [];
	const maxMiter = inset * 2.4;
	for (let i = 0; i < n; i++) {
		const prev = lines[(i - 1 + n) % n];
		const curr = lines[i];
		const det = prev.dx * curr.dy - prev.dy * curr.dx;
		if (Math.abs(det) < 1e-4) {
			out.push({ x: curr.px, y: curr.py });
			continue;
		}
		const rx = curr.px - prev.px;
		const ry = curr.py - prev.py;
		const t = (rx * curr.dy - ry * curr.dx) / det;
		let ix = prev.px + prev.dx * t;
		let iy = prev.py + prev.dy * t;
		const orig = ccw[i];
		const dist = Math.hypot(ix - orig.x, iy - orig.y);
		if (dist > maxMiter) {
			const k = maxMiter / dist;
			ix = orig.x + (ix - orig.x) * k;
			iy = orig.y + (iy - orig.y) * k;
		}
		out.push({ x: ix, y: iy });
	}
	return out;
}

/** Create a rounded THREE.Shape from a CCW polygon in world coordinates. */
function buildRoundedShape(pts: Vec2[], radius: number) {
	const shape = new THREE.Shape();
	const n = pts.length;
	for (let i = 0; i < n; i++) {
		const prev = pts[(i - 1 + n) % n];
		const curr = pts[i];
		const next = pts[(i + 1) % n];
		const v1x = prev.x - curr.x;
		const v1y = prev.y - curr.y;
		const v2x = next.x - curr.x;
		const v2y = next.y - curr.y;
		const len1 = Math.hypot(v1x, v1y) || 1;
		const len2 = Math.hypot(v2x, v2y) || 1;
		const r = Math.min(radius, len1 * 0.34, len2 * 0.34);
		const p1x = curr.x + (v1x / len1) * r;
		const p1y = curr.y + (v1y / len1) * r;
		const p2x = curr.x + (v2x / len2) * r;
		const p2y = curr.y + (v2y / len2) * r;

		if (i === 0) {
			shape.moveTo(p1x, p1y);
		} else {
			shape.lineTo(p1x, p1y);
		}
		shape.quadraticCurveTo(curr.x, curr.y, p2x, p2y);
	}
	shape.closePath();
	return shape;
}

/** Build a rounded rectangle centered at (0, 0). */
function buildRoundedRectShape(w: number, h: number, radius: number) {
	const hw = w / 2;
	const hh = h / 2;
	const r = Math.min(radius, hw * 0.4, hh * 0.4);
	const s = new THREE.Shape();
	s.moveTo(-hw + r, -hh);
	s.lineTo(hw - r, -hh);
	s.quadraticCurveTo(hw, -hh, hw, -hh + r);
	s.lineTo(hw, hh - r);
	s.quadraticCurveTo(hw, hh, hw - r, hh);
	s.lineTo(-hw + r, hh);
	s.quadraticCurveTo(-hw, hh, -hw, hh - r);
	s.lineTo(-hw, -hh + r);
	s.quadraticCurveTo(-hw, -hh, -hw + r, -hh);
	return s;
}

/** Compute polygon centroid. */
function polygonCentroid(pts: Vec2[]): Vec2 {
	let sx = 0;
	let sy = 0;
	for (const p of pts) {
		sx += p.x;
		sy += p.y;
	}
	return { x: sx / pts.length, y: sy / pts.length };
}

/** Mean Value Coordinates (Floater 2003) for smooth, zero-allocation 2D cage morphing. */
function computeMVC(qx: number, qy: number, poly: Vec2[]): Float32Array {
	const k = poly.length;
	const w = new Float32Array(k);
	const dx = new Float64Array(k);
	const dy = new Float64Array(k);
	const r = new Float64Array(k);

	for (let i = 0; i < k; i++) {
		dx[i] = poly[i].x - qx;
		dy[i] = poly[i].y - qy;
		r[i] = Math.hypot(dx[i], dy[i]);
		if (r[i] < 1e-4) {
			w[i] = 1;
			return w;
		}
	}

	const tanHalf = new Float64Array(k);
	for (let i = 0; i < k; i++) {
		const j = (i + 1) % k;
		const dot = dx[i] * dx[j] + dy[i] * dy[j];
		const cross = dx[i] * dy[j] - dy[i] * dx[j];
		const denom = r[i] * r[j] + dot;
		if (Math.abs(cross) < 1e-4 * r[i] * r[j] && dot < 0) {
			const sumR = r[i] + r[j];
			w[i] = r[j] / sumR;
			w[j] = r[i] / sumR;
			return w;
		}
		tanHalf[i] = denom > 1e-8 ? cross / denom : 0;
	}

	let sumW = 0;
	for (let i = 0; i < k; i++) {
		const prev = (i - 1 + k) % k;
		const wi = (tanHalf[prev] + tanHalf[i]) / r[i];
		w[i] = wi;
		sumW += wi;
	}

	if (Math.abs(sumW) > 1e-8) {
		const inv = 1 / sumW;
		for (let i = 0; i < k; i++) {
			w[i] *= inv;
		}
	} else {
		w.fill(1 / k);
	}
	return w;
}

/** Bake flat MVC weight table once per geometry so runtime morphing does zero allocations. */
function bakeMorphWeights(
	// biome-ignore lint/suspicious/noExplicitAny: three.js BufferGeometry
	geo: any,
	cx: number,
	cy: number,
	poly: Vec2[],
	sources?: Uint32Array,
) {
	const posAttr = geo.attributes.position;
	const count: number = posAttr.count;
	const k = poly.length;
	const table = new Float32Array(count * k);
	// Extruded triangles repeat XY at seams and at different depths. Their cage motion is identical.
	const firstAt = sources && new Map<string, number>();
	for (let v = 0; v < count; v++) {
		const wx = posAttr.getX(v) + cx;
		const wy = posAttr.getY(v) + cy;
		if (firstAt && sources) {
			const key = `${wx},${wy}`;
			const previous = firstAt.get(key);
			sources[v] = previous ?? v;
			if (previous !== undefined) continue;
			firstAt.set(key, v);
		}
		const weights = computeMVC(wx, wy, poly);
		const offset = v * k;
		for (let j = 0; j < k; j++) {
			table[offset + j] = weights[j];
		}
	}
	posAttr.setUsage(THREE.DynamicDrawUsage);
	return table;
}

/** Fast 2D analytic point-in-polygon ray test (~0.002ms). */
function pointInPolygon(px: number, py: number, poly: Vec2[]): boolean {
	let inside = false;
	for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
		const xi = poly[i].x;
		const yi = poly[i].y;
		const xj = poly[j].x;
		const yj = poly[j].y;
		const intersect =
			yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi;
		if (intersect) inside = !inside;
	}
	return inside;
}

/** Solve exact 2D projective homography and return CSS matrix3d string. */
function solveMatrix3d(w: number, h: number, dst: Vec2[]): string | null {
	// The four source points are always the corners of an axis-aligned rectangle.
	// Solve the two perspective terms on the unit square, then scale its columns.
	const [tl, tr, br, bl] = dst;
	const dx1 = tr.x - br.x;
	const dy1 = tr.y - br.y;
	const dx2 = bl.x - br.x;
	const dy2 = bl.y - br.y;
	const sx = tl.x - tr.x + br.x - bl.x;
	const sy = tl.y - tr.y + br.y - bl.y;
	const det = dx1 * dy2 - dx2 * dy1;
	if (Math.abs(det) < 1e-10) return null;
	const g = (sx * dy2 - dx2 * sy) / det;
	const k = (dx1 * sy - sx * dy1) / det;
	const m = [
		(tr.x - tl.x + g * tr.x) / w,
		(tr.y - tl.y + g * tr.y) / w,
		0,
		g / w,
		(bl.x - tl.x + k * bl.x) / h,
		(bl.y - tl.y + k * bl.y) / h,
		0,
		k / h,
		0,
		0,
		1,
		0,
		tl.x,
		tl.y,
		0,
		1,
	];
	if (m.some((v) => !Number.isFinite(v))) return null;
	return `matrix3d(${m.map((v) => v.toFixed(7)).join(",")})`;
}

/** Wrap mixed CJK + Latin copy cleanly within `maxWidth` and return final baseline Y. */
function drawWrappedText(
	ctx: CanvasRenderingContext2D,
	text: string,
	x: number,
	startY: number,
	maxWidth: number,
	lineHeight: number,
	maxLines = 3,
): number {
	const lines: string[] = [];
	let current = "";
	for (const ch of text) {
		if (lines.length >= maxLines - 1) {
			current += ch;
			continue;
		}
		const next = current + ch;
		if (ctx.measureText(next).width <= maxWidth || current.length === 0) {
			current = next;
		} else {
			const lastSpace = current.lastIndexOf(" ");
			if (lastSpace > current.length * 0.45 && /[A-Za-z0-9]/.test(ch)) {
				lines.push(current.slice(0, lastSpace).trimEnd());
				current = current.slice(lastSpace + 1) + ch;
			} else {
				lines.push(current.trimEnd());
				current = ch.trimStart();
			}
		}
	}
	if (current.trim().length > 0 && lines.length < maxLines) {
		let lastLine = current.trim();
		while (lastLine.length > 1 && ctx.measureText(lastLine).width > maxWidth) {
			lastLine = `${lastLine.slice(0, -2)}…`;
		}
		lines.push(lastLine);
	}
	for (let i = 0; i < lines.length; i++) {
		ctx.fillText(lines[i], x, startY + i * lineHeight);
	}
	return startY + Math.max(0, lines.length - 1) * lineHeight;
}

/** Paint the crisp HiDPI top nav crystal slab canvas. */
function renderNavCanvas(
	w: number,
	h: number,
	dpr: number,
	isDark: boolean,
	accent: string,
	meta: GateMeta,
	isPortrait: boolean,
): HTMLCanvasElement {
	const canvas = document.createElement("canvas");
	canvas.width = Math.max(2, Math.round(w * dpr));
	canvas.height = Math.max(2, Math.round(h * dpr));
	const ctx = canvas.getContext("2d");
	if (!ctx) return canvas;

	ctx.scale(dpr, dpr);
	ctx.clearRect(0, 0, w, h);

	ctx.fillStyle = isDark
		? "rgba(11, 8, 17, 0.56)"
		: "rgba(255, 250, 253, 0.36)";
	ctx.fillRect(0, 0, w, h);

	const padX = isPortrait ? w * 0.055 : w * 0.036;
	const cy = h * 0.5;
	const ink = isDark ? "rgba(252,248,251,0.95)" : "rgba(28,18,30,0.92)";
	const muted = isDark ? "rgba(244,234,242,0.68)" : "rgba(42,26,44,0.62)";
	const line = isDark ? "rgba(255,255,255,0.14)" : "rgba(28,18,30,0.12)";

	const topLineGrad = ctx.createLinearGradient(0, 0, w, 0);
	topLineGrad.addColorStop(0, "transparent");
	topLineGrad.addColorStop(
		0.5,
		isDark ? "rgba(255,255,255,0.22)" : "rgba(255,255,255,0.88)",
	);
	topLineGrad.addColorStop(1, "transparent");
	ctx.fillStyle = topLineGrad;
	ctx.fillRect(0, 0, w, 1.2);

	const markR = clamp(h * 0.2, 8, 14);
	const mx = padX + markR;
	ctx.save();
	ctx.translate(mx, cy);
	ctx.strokeStyle = accent;
	ctx.lineWidth = 1.4;
	ctx.beginPath();
	ctx.rect(-markR * 0.72, -markR * 0.72, markR * 1.44, markR * 1.44);
	ctx.stroke();
	ctx.rotate(Math.PI / 4);
	ctx.strokeStyle = ink;
	ctx.globalAlpha = 0.65;
	ctx.beginPath();
	ctx.rect(-markR * 0.68, -markR * 0.68, markR * 1.36, markR * 1.36);
	ctx.stroke();
	ctx.globalAlpha = 1;
	ctx.fillStyle = accent;
	ctx.beginPath();
	ctx.arc(0, 0, markR * 0.22, 0, Math.PI * 2);
	ctx.fill();
	ctx.restore();

	const titleSize = clamp(h * 0.25, 13, 18);
	ctx.fillStyle = ink;
	ctx.font = `600 ${titleSize}px "PingFang SC", "Roboto", system-ui, sans-serif`;
	ctx.textBaseline = "middle";
	ctx.textAlign = "left";
	ctx.fillText(meta.siteTitle, mx + markR + 12, cy + 0.5);

	// Right divider before DOM audio + skip controls
	if (!isPortrait && w >= 720) {
		const rightInfoX = w - padX - 152;
		ctx.strokeStyle = line;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(rightInfoX + 14, cy - h * 0.16);
		ctx.lineTo(rightInfoX + 14, cy + h * 0.16);
		ctx.stroke();

		const badgeSize = clamp(h * 0.165, 10, 12);
		ctx.font = `500 ${badgeSize}px "JetBrains Mono Variable", "PingFang SC", ui-monospace, monospace`;
		ctx.fillStyle = muted;
		ctx.textAlign = "right";
		ctx.fillText(
			`${meta.postCount} 篇笔记 · ${meta.topicCount} 个专题`,
			rightInfoX,
			cy + 0.5,
		);
	}

	return canvas;
}

interface PanelRenderResult {
	canvas: HTMLCanvasElement;
	ctaTopPct: number;
	sealTopPct: number;
	sealNormX: number;
	sealNormY: number;
	sealRadiusPx: number;
}

/** Paint the crisp HiDPI main crystal panel editorial cover artwork & etched horology blueprint dial. */
function renderPanelCanvas(
	w: number,
	h: number,
	dpr: number,
	isDark: boolean,
	accent: string,
	accentSoft: string,
	hue: number,
	meta: GateMeta,
	isPortrait: boolean,
): PanelRenderResult {
	const canvas = document.createElement("canvas");
	canvas.width = Math.max(2, Math.round(w * dpr));
	canvas.height = Math.max(2, Math.round(h * dpr));
	const ctx = canvas.getContext("2d");
	const sealNormX = isPortrait ? 0.5 : 0.805;
	const sealNormY = isPortrait ? 0.765 : 0.455;
	const sealR = isPortrait
		? clamp(Math.min(w * 0.22, h * 0.12), 58, 92)
		: clamp(Math.min(w * 0.145, h * 0.3), 92, 180);

	if (!ctx) {
		return {
			canvas,
			ctaTopPct: isPortrait ? 40 : 61,
			sealTopPct: sealNormY * 100,
			sealNormX,
			sealNormY,
			sealRadiusPx: sealR,
		};
	}

	ctx.scale(dpr, dpr);
	ctx.clearRect(0, 0, w, h);

	// Smoked-glass / frosted porcelain face veil
	ctx.fillStyle = isDark
		? "rgba(10, 7, 16, 0.56)"
		: "rgba(255, 251, 253, 0.36)";
	ctx.fillRect(0, 0, w, h);

	const ink = isDark ? "#fbf7fa" : "#1c131d";
	const inkSub = isDark ? "rgba(246,238,244,0.78)" : "rgba(32,20,34,0.68)";
	const inkMuted = isDark ? "rgba(244,235,242,0.50)" : "rgba(32,20,34,0.46)";
	const hairline = isDark ? "rgba(255,255,255,0.09)" : "rgba(32,20,34,0.09)";
	const padX = isPortrait ? w * 0.072 : w * 0.068;
	const maxTextW = isPortrait ? w - padX * 2 : w * 0.495;

	const sealCx = w * sealNormX;
	const sealCy = h * sealNormY;

	const wash = ctx.createRadialGradient(
		sealCx,
		sealCy,
		10,
		sealCx,
		sealCy,
		Math.max(w, h) * 0.56,
	);
	wash.addColorStop(
		0,
		isDark ? `hsla(${hue}, 82%, 64%, 0.16)` : `hsla(${hue}, 78%, 64%, 0.12)`,
	);
	wash.addColorStop(1, "transparent");
	ctx.fillStyle = wash;
	ctx.fillRect(0, 0, w, h);

	// Architectural drafting hairlines + subtle coordinate crosshairs
	ctx.strokeStyle = hairline;
	ctx.lineWidth = 1;
	const footerY = h * (isPortrait ? 0.94 : 0.905);
	ctx.beginPath();
	ctx.moveTo(padX, footerY - h * 0.036);
	ctx.lineTo(w - padX, footerY - h * 0.036);
	ctx.stroke();

	if (!isPortrait) {
		ctx.beginPath();
		ctx.moveTo(w * 0.618, h * 0.08);
		ctx.lineTo(w * 0.618, footerY - h * 0.055);
		ctx.stroke();
	}

	const drawCross = (cx: number, cy: number) => {
		ctx.strokeStyle = inkMuted;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(cx - 5, cy);
		ctx.lineTo(cx + 5, cy);
		ctx.moveTo(cx, cy - 5);
		ctx.lineTo(cx, cy + 5);
		ctx.stroke();
	};
	drawCross(padX * 0.52, h * 0.055);
	drawCross(w - padX * 0.52, h * 0.055);
	drawCross(padX * 0.52, h * 0.945);
	drawCross(w - padX * 0.52, h * 0.945);

	// Giant subtle watermark ghost glyph behind the 3D crystal astrolabe
	ctx.save();
	const ghostSize = isPortrait ? w * 0.36 : Math.min(w * 0.22, h * 0.52);
	ctx.font = `700 ${ghostSize}px "Songti SC", "Noto Serif SC", "STSong", Georgia, serif`;
	ctx.fillStyle = isDark ? "rgba(255,255,255,0.032)" : "rgba(28,18,30,0.036)";
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	ctx.fillText("格物", sealCx, sealCy);
	ctx.restore();

	// ── Editorial Left / Top Hero Typography ─────────────────────────
	const eyebrowY = h * (isPortrait ? 0.068 : 0.145);
	const eyebrowSize = clamp(isPortrait ? w * 0.027 : w * 0.009, 9.5, 13);

	ctx.fillStyle = accent;
	ctx.fillRect(padX, eyebrowY - 1, 20, 2);

	ctx.font = `600 ${eyebrowSize}px "JetBrains Mono Variable", ui-monospace, monospace`;
	ctx.fillStyle = accent;
	ctx.textAlign = "left";
	ctx.textBaseline = "middle";
	ctx.fillText("MATH · ML · DL · LLM", padX + 28, eyebrowY);

	const h1Size = isPortrait
		? clamp(w * 0.104, 30, 48)
		: clamp(Math.min(w * 0.06, h * 0.135), 40, 84);
	const h1LineH = h1Size * 1.14;
	const h1StartY = eyebrowY + h1Size * 0.88;

	const h1Grad = ctx.createLinearGradient(
		0,
		h1StartY - h1Size * 0.5,
		0,
		h1StartY + h1LineH * 1.4,
	);
	if (isDark) {
		h1Grad.addColorStop(0, "#ffffff");
		h1Grad.addColorStop(0.65, "#f3ebf2");
		h1Grad.addColorStop(1, "#d5cad5");
	} else {
		h1Grad.addColorStop(0, "#140c16");
		h1Grad.addColorStop(0.7, "#2a1b2d");
		h1Grad.addColorStop(1, "#4a334d");
	}

	ctx.font = `600 ${h1Size}px "Songti SC", "Noto Serif SC", "Source Han Serif SC", "STSong", "PingFang SC", Georgia, serif`;
	ctx.fillStyle = h1Grad;
	ctx.textAlign = "left";
	ctx.textBaseline = "middle";
	ctx.fillText("吾生也有涯，", padX, h1StartY);
	ctx.fillText("而知也无涯", padX, h1StartY + h1LineH);

	const line2Width = ctx.measureText("而知也无涯").width;
	ctx.fillStyle = accent;
	ctx.fillText("。", padX + line2Width - h1Size * 0.04, h1StartY + h1LineH);

	const epigraphSize = clamp(isPortrait ? w * 0.032 : w * 0.0125, 11, 17);
	const epigraphY = h1StartY + h1LineH * 1.68;
	ctx.font = `italic 400 ${epigraphSize}px Georgia, "Times New Roman", serif`;
	ctx.fillStyle = inkMuted;
	const epigraphEndY = drawWrappedText(
		ctx,
		"Notes on mathematics, machine learning, deep learning, and LLMs.",
		padX,
		epigraphY,
		maxTextW,
		epigraphSize * 1.42,
		2,
	);

	const subSize = clamp(isPortrait ? w * 0.033 : w * 0.0125, 12, 16.5);
	const subLineH = subSize * 1.58;
	const subStartY = epigraphEndY + subSize * 1.75;
	ctx.font = `400 ${subSize}px "PingFang SC", "Roboto", system-ui, sans-serif`;
	ctx.fillStyle = inkSub;
	const sub1EndY = drawWrappedText(
		ctx,
		"聚焦数学、机器学习、深度学习与 LLM。",
		padX,
		subStartY,
		maxTextW,
		subLineH,
		2,
	);
	const sub2EndY = drawWrappedText(
		ctx,
		`共收录 ${meta.postCount} 篇笔记，按专题与时间归档。`,
		padX,
		sub1EndY + subLineH,
		maxTextW,
		subLineH,
		2,
	);

	const ctaGap = isPortrait ? 18 : 26;
	const rawCtaTopPct = ((sub2EndY + ctaGap) / h) * 100;
	const ctaTopPct = isPortrait
		? clamp(rawCtaTopPct, 36, 43)
		: clamp(rawCtaTopPct, 56, 65);

	if (!isPortrait) {
		const metricsY = Math.max(h * 0.785, (ctaTopPct / 100) * h + 74);
		const pad2 = (n: number) => String(n).padStart(2, "0");
		const stats = [
			{ val: pad2(meta.postCount), lbl: "文章" },
			{ val: pad2(meta.topicCount), lbl: "专题" },
			{ val: pad2(meta.tagCount), lbl: "标签" },
		];
		const colStep = clamp(w * 0.142, 108, 172);
		for (let i = 0; i < stats.length; i++) {
			const sx = padX + i * colStep;
			ctx.font = `300 ${clamp(w * 0.024, 21, 32)}px "JetBrains Mono Variable", ui-monospace, monospace`;
			ctx.fillStyle = ink;
			ctx.fillText(stats[i].val, sx, metricsY);

			ctx.font = `500 ${clamp(w * 0.0085, 10, 11.5)}px "PingFang SC", "Roboto", sans-serif`;
			ctx.fillStyle = inkMuted;
			ctx.fillText(stats[i].lbl, sx, metricsY + 22);
		}
	}

	// ── Etched Precision Blueprint Dial (sits underneath the real 3D gyroscopic prism!) ──
	ctx.save();
	ctx.translate(sealCx, sealCy);

	for (let i = 0; i < 72; i++) {
		const ang = (i / 72) * Math.PI * 2;
		const major = i % 6 === 0;
		const r0 = sealR * (major ? 0.91 : 0.96);
		const r1 = sealR;
		ctx.strokeStyle = major ? accentSoft : hairline;
		ctx.lineWidth = major ? 1.3 : 0.8;
		ctx.beginPath();
		ctx.moveTo(Math.cos(ang) * r0, Math.sin(ang) * r0);
		ctx.lineTo(Math.cos(ang) * r1, Math.sin(ang) * r1);
		ctx.stroke();
	}

	ctx.strokeStyle = hairline;
	ctx.lineWidth = 1;
	ctx.beginPath();
	ctx.arc(0, 0, sealR * 0.85, 0, Math.PI * 2);
	ctx.stroke();

	ctx.setLineDash([3, 7]);
	ctx.strokeStyle = accentSoft;
	ctx.beginPath();
	ctx.arc(0, 0, sealR * 0.72, 0, Math.PI * 2);
	ctx.stroke();
	ctx.setLineDash([]);

	// Subtle blueprint crosshairs centered behind the 3D floating crystal
	ctx.strokeStyle = hairline;
	ctx.lineWidth = 0.8;
	ctx.beginPath();
	ctx.moveTo(-sealR * 1.08, 0);
	ctx.lineTo(-sealR * 0.88, 0);
	ctx.moveTo(sealR * 0.88, 0);
	ctx.lineTo(sealR * 1.08, 0);
	ctx.moveTo(0, -sealR * 1.08);
	ctx.lineTo(0, -sealR * 0.88);
	ctx.moveTo(0, sealR * 0.88);
	ctx.lineTo(0, sealR * 1.08);
	ctx.stroke();

	if (!isPortrait) {
		ctx.font = `500 ${clamp(w * 0.0078, 9.5, 11)}px "PingFang SC", "JetBrains Mono Variable", sans-serif`;
		ctx.fillStyle = inkMuted;
		ctx.textAlign = "center";
		ctx.fillText(meta.siteTitle, 0, -sealR - 18);
		ctx.fillText("点击进入", 0, sealR + 22);
	}

	ctx.restore();

	const footSize = clamp(isPortrait ? w * 0.025 : w * 0.0082, 9.5, 12);
	ctx.font = `500 ${footSize}px "JetBrains Mono Variable", "PingFang SC", ui-monospace, monospace`;
	ctx.fillStyle = inkMuted;
	ctx.textBaseline = "middle";

	ctx.textAlign = "left";
	ctx.fillText("© CHENHUI · XUCHENHUI.CC", padX, footerY);

	if (!isPortrait && w >= 920) {
		ctx.textAlign = "center";
		ctx.fillText("下滑或按 Enter 进入", w * 0.5, footerY);
	}

	ctx.textAlign = "right";
	ctx.fillText(`更新于 ${meta.lastUpdated}`, w - padX, footerY);

	return {
		canvas,
		ctaTopPct,
		sealTopPct: sealNormY * 100,
		sealNormX,
		sealNormY,
		sealRadiusPx: sealR,
	};
}

// ── Custom GLSL Shaders ──────────────────────────────────────────────

/** Real-time Domain-Warped Liquid Caustic Silk backdrop glowing through the crystal gaps. */
const CAUSTIC_VERT = `
varying vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const CAUSTIC_FRAG = `
precision highp float;
uniform float uTime;
uniform vec2 uPointer;
uniform vec2 uVelocity;
uniform vec3 uBaseA;
uniform vec3 uBaseB;
uniform vec3 uAccent;
uniform vec3 uAccentAlt;
uniform float uDark;
uniform float uBurst;
varying vec2 vUv;

void main() {
	vec2 p = vUv * 2.0 - 1.0;
	float t = uTime * 0.28;

	// Lightweight 3-octave trigonometric curl domain warping (super fast, ultra silky!)
	vec2 q = p * 1.85;
	vec2 ptr = uPointer * 1.15;
	float ptrDist = length(p - ptr);
	float wake = exp(-ptrDist * ptrDist * 2.8);
	q += uVelocity * wake * 1.6;

	for (int i = 0; i < 3; i++) {
		float fi = float(i) + 1.0;
		q += vec2(
			sin(q.y * 1.65 * fi + t * (1.1 + 0.3 * fi)),
			cos(q.x * 1.55 * fi - t * (0.9 + 0.25 * fi))
		) * (0.34 / fi);
	}

	// Caustic interference filaments
	float wave = sin(q.x * 2.4 + q.y * 2.1 - t * 1.5);
	float caustic = pow(1.0 - abs(wave), 4.0);
	float silk = 0.5 + 0.5 * sin(q.x * 1.4 - q.y * 1.2 + t);

	vec3 col = mix(uBaseA, uBaseB, silk);
	// Iridescent chromatic caustic ribbons visible through crystal gaps
	vec3 ribbonCol = mix(uAccent, uAccentAlt, 0.5 + 0.5 * sin(q.y * 2.2 + t * 0.9));
	float ribbonStrength = mix(0.24, 0.42, uDark);
	col += ribbonCol * caustic * ribbonStrength;
	col += uAccent * wake * mix(0.14, 0.28, uDark);

	// Soft edge vignette & unseal fade
	float vig = 1.0 - dot(p * 0.55, p * 0.55);
	col *= clamp(vig + 0.35, 0.55, 1.08);
	float alpha = clamp(1.0 - uBurst * 1.45, 0.0, 1.0);
	gl_FragColor = vec4(col, alpha);
}
`;

/** 100% GPU Vertex/Fragment Shader Bokeh & 4-Point Diamond Glint Vortex Particle Cloud. */
const VORTEX_VERT = `
attribute vec3 aSeed;
uniform float uTime;
uniform float uDpr;
uniform vec2 uPtrWorld;
uniform float uPtrActive;
uniform float uBurst;
uniform vec2 uBurstOrigin;
varying float vAlpha;
varying float vSpark;
varying float vMix;

void main() {
	vec3 pos = position;
	float ph = aSeed.x * 6.2831;
	float spd = 0.35 + aSeed.y * 0.45;

	// Organic Lissajous drift
	pos.x += sin(uTime * spd + ph) * (24.0 + aSeed.z * 22.0);
	pos.y += cos(uTime * spd * 0.82 + ph * 1.3) * (20.0 + aSeed.y * 20.0);

	// Real-time GPU cursor vortex swirl wake
	vec2 toPtr = pos.xy - uPtrWorld;
	float d = length(toPtr);
	float swirlInf = exp(-d * d / 56000.0) * uPtrActive * (1.0 - uBurst);
	vec2 tangent = vec2(-toPtr.y, toPtr.x) / max(d, 1.0);
	pos.xy += tangent * swirlInf * 68.0 + normalize(toPtr + vec2(0.001)) * swirlInf * 24.0;

	// Radial spiral explosion on unseal (100% GPU!)
	if (uBurst > 0.001) {
		vec2 outDir = normalize(pos.xy - uBurstOrigin + vec2(0.01));
		float angle = uBurst * (1.6 + aSeed.z * 1.8);
		float ca = cos(angle);
		float sa = sin(angle);
		vec2 rotDir = vec2(outDir.x * ca - outDir.y * sa, outDir.x * sa + outDir.y * ca);
		float ease = uBurst * uBurst;
		pos.xy += rotDir * ease * (460.0 + aSeed.y * 420.0);
		pos.z += ease * (320.0 + aSeed.x * 260.0);
	}

	vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
	gl_Position = projectionMatrix * mvPosition;

	float twinkle = 0.55 + 0.45 * sin(uTime * (2.2 + aSeed.y * 2.8) + ph * 2.0);
	float baseSize = mix(10.0, 26.0, pow(aSeed.z, 2.2));
	gl_PointSize = baseSize * uDpr * (0.8 + 0.45 * twinkle + swirlInf * 0.7) * (1200.0 / max(-mvPosition.z, 200.0));

	vAlpha = (0.34 + 0.42 * twinkle + swirlInf * 0.35) * (1.0 - smoothstep(0.45, 1.0, uBurst));
	vSpark = step(0.68, aSeed.z);
	vMix = aSeed.y;
}
`;

const VORTEX_FRAG = `
precision highp float;
uniform vec3 uColA;
uniform vec3 uColB;
varying float vAlpha;
varying float vSpark;
varying float vMix;

void main() {
	vec2 uv = gl_PointCoord * 2.0 - 1.0;
	float r = length(uv);
	if (r > 1.0) discard;

	// Soft circular optical bokeh disc
	float disc = smoothstep(1.0, 0.08, r);
	float core = smoothstep(0.45, 0.0, r);

	// 4-point diamond star glint on highlight particles
	float star = max(
		exp(-abs(uv.x) * 9.0) * exp(-abs(uv.y) * 2.2),
		exp(-abs(uv.y) * 9.0) * exp(-abs(uv.x) * 2.2)
	);
	float shape = mix(disc * 0.75 + core * 0.45, max(core * 0.9, star * 1.15), vSpark);

	vec3 col = mix(uColA, uColB, vMix);
	gl_FragColor = vec4(col, shape * vAlpha);
}
`;

export function initEntranceGate(
	root: HTMLElement,
	meta: GateMeta,
): () => void {
	const canvas = root.querySelector<HTMLCanvasElement>("#eg-canvas");
	const backdrop = root.querySelector<HTMLElement>("#eg-backdrop");
	const fallbackEl = root.querySelector<HTMLElement>("#eg-fallback");
	const panelOverlay = root.querySelector<HTMLElement>("#eg-panel-overlay");
	const navOverlay = root.querySelector<HTMLElement>("#eg-nav-overlay");
	const shockwaveEl = root.querySelector<HTMLElement>("#eg-shockwave");
	const openBtns = root.querySelectorAll<HTMLElement>("[data-eg-open]");
	const sealHitBtn = root.querySelector<HTMLElement>("#eg-seal-hit");
	const magneticBtn = root.querySelector<HTMLElement>("#eg-cta-ring");
	const telemetryEl = root.querySelector<HTMLElement>("#eg-telemetry-text");
	const audioBtn = root.querySelector<HTMLButtonElement>("#eg-audio-btn");
	const reticleEl = root.querySelector<HTMLElement>("#eg-reticle");

	const reduceMotion = window.matchMedia(
		"(prefers-reduced-motion: reduce)",
	).matches;
	const isCoarsePointer = window.matchMedia("(pointer: coarse)").matches;
	const audioSynth = new CrystalAudioSynth();

	let disposed = false;
	let phase: "intro" | "idle" | "shattering" | "done" = "intro";
	let phaseStart = performance.now();
	let pendingRoute: string | null = null;
	let rafId = 0;

	// Live Scramble-Decode Telemetry controller (only animates when hovered shard changes!)
	let activeTelemetryIdx = -1;
	let scrambleTimer: ReturnType<typeof setInterval> | undefined;
	const triggerScramble = (idx: number) => {
		if (!telemetryEl || idx === activeTelemetryIdx) return;
		activeTelemetryIdx = idx;
		const target = SHARD_TELEMETRY[idx] ?? SHARD_TELEMETRY[0];
		clearInterval(scrambleTimer);
		if (reduceMotion) {
			telemetryEl.textContent = target;
			return;
		}
		let step = 0;
		const totalSteps = 12;
		scrambleTimer = setInterval(() => {
			if (disposed || !telemetryEl) {
				clearInterval(scrambleTimer);
				return;
			}
			step++;
			const revealCount = Math.floor((step / totalSteps) * target.length);
			let out = "";
			for (let i = 0; i < target.length; i++) {
				const ch = target[i];
				if (i < revealCount || ch === " " || ch === "/" || ch === "·") {
					out += ch;
				} else {
					out +=
						SCRAMBLE_GLYPHS[Math.floor(Math.random() * SCRAMBLE_GLYPHS.length)];
				}
			}
			telemetryEl.textContent = out;
			if (step >= totalSteps) {
				telemetryEl.textContent = target;
				clearInterval(scrambleTimer);
			}
		}, 24);
	};

	const onAudioToggle = () => {
		const on = audioSynth.toggle();
		if (audioBtn) {
			audioBtn.dataset.active = on ? "true" : "false";
			audioBtn.setAttribute("aria-label", on ? "关闭晶体音效" : "开启晶体音效");
		}
	};
	if (audioBtn) {
		audioBtn.addEventListener("click", onAudioToggle);
	}

	const finishAndCleanup = () => {
		if (disposed) return;
		disposed = true;
		phase = "done";
		clearInterval(scrambleTimer);
		if (rafId) cancelAnimationFrame(rafId);
		document.documentElement.classList.remove(
			"has-entrance-gate",
			"gate-revealing",
		);
		root.setAttribute("aria-hidden", "true");
		root.style.display = "none";

		try {
			const u = new URL(window.location.href);
			if (u.searchParams.has("gate") || u.searchParams.has("cover")) {
				u.searchParams.delete("gate");
				u.searchParams.delete("cover");
				window.history.replaceState({}, "", u.toString());
			}
		} catch (_) {}

		disposeThree();
		audioSynth.dispose();

		if (pendingRoute) {
			const target = pendingRoute;
			pendingRoute = null;
			// biome-ignore lint/suspicious/noExplicitAny: Swup runtime navigate API
			const swupAny = window.swup as any;
			if (swupAny && typeof swupAny.navigate === "function") {
				swupAny.navigate(target);
			} else {
				window.location.href = target;
			}
		}
	};

	const markSeen = () => {
		try {
			localStorage.setItem(STORAGE_KEY, "1");
		} catch (_) {}
	};

	const spawnShockwave = (cx: number, cy: number) => {
		if (!shockwaveEl || reduceMotion) return;
		shockwaveEl.style.left = `${cx}px`;
		shockwaveEl.style.top = `${cy}px`;
		shockwaveEl.classList.add("is-burst");
	};

	let webglReady = false;
	const triggerUnseal = (
		originX?: number,
		originY?: number,
		routeUrl?: string,
	) => {
		if (phase === "shattering" || phase === "done") return;
		markSeen();
		pendingRoute = routeUrl || null;
		phase = "shattering";
		phaseStart = performance.now();

		const ox = originX ?? window.innerWidth * 0.5;
		const oy = originY ?? window.innerHeight * 0.5;
		spawnShockwave(ox, oy);
		audioSynth.playUnsealChord();

		root.classList.add("is-shattering");
		document.documentElement.classList.add("gate-revealing");
		document.documentElement.classList.remove("has-entrance-gate");

		if (!webglReady || reduceMotion) {
			if (backdrop) backdrop.style.opacity = "0";
			if (fallbackEl) fallbackEl.style.opacity = "0";
			setTimeout(finishAndCleanup, reduceMotion ? 160 : 480);
			return;
		}

		// Trigger GLSL caustic ripple & GPU particle burst from exact click coordinates
		const worldClickX = ox - W * 0.5;
		const worldClickY = H * 0.5 - oy;
		sharedGlassUniforms.uRippleOrigin.value.set(worldClickX, worldClickY);
		sharedGlassUniforms.uRippleAge.value = 0.001;
		if (vortexUniforms) {
			vortexUniforms.uBurstOrigin.value.set(worldClickX, worldClickY);
		}

		for (const p of pieces) {
			p.shadowMesh.visible = false;
			if (p.isNav) {
				p.flyDelay = 0;
			} else {
				const dx = p.baseX - worldClickX;
				const dy = p.baseY - worldClickY;
				const dist = Math.hypot(dx, dy);
				p.flyDelay = 0.015 + clamp(dist / Math.max(W, H), 0, 1) * 0.065;
			}
		}

		setTimeout(() => {
			if (disposed) return;
			if (backdrop) backdrop.style.opacity = "0";
		}, 110);

		setTimeout(() => {
			if (!disposed && canvas) {
				canvas.style.transition = "opacity 280ms ease-out";
				canvas.style.opacity = "0";
			}
		}, 390);

		setTimeout(finishAndCleanup, 720);
	};

	const onOpenClick = (e: Event) => {
		const me = e as MouseEvent;
		triggerUnseal(
			me.clientX || window.innerWidth * 0.5,
			me.clientY || window.innerHeight * 0.5,
		);
	};
	for (const btn of openBtns) {
		btn.addEventListener("click", onOpenClick);
	}

	let isSealHovered = false;
	const onSealEnter = () => {
		isSealHovered = true;
		reticleEl?.classList.add("is-locked");
		triggerScramble(5);
		audioSynth.playHoverChime(5);
	};
	const onSealLeave = () => {
		isSealHovered = false;
		reticleEl?.classList.remove("is-locked");
	};
	const onSealClick = (e: Event) => {
		const me = e as MouseEvent;
		triggerUnseal(me.clientX, me.clientY);
	};
	if (sealHitBtn) {
		sealHitBtn.addEventListener("click", onSealClick);
		sealHitBtn.addEventListener("pointerenter", onSealEnter);
		sealHitBtn.addEventListener("pointerleave", onSealLeave);
	}

	const onCtaEnter = () => reticleEl?.classList.add("is-locked");
	const onCtaLeave = () => reticleEl?.classList.remove("is-locked");
	if (magneticBtn) {
		magneticBtn.addEventListener("pointerenter", onCtaEnter);
		magneticBtn.addEventListener("pointerleave", onCtaLeave);
	}

	const onKeyDown = (e: KeyboardEvent) => {
		if (phase === "shattering" || phase === "done") return;
		if (e.key === "Enter" || e.key === " " || e.key === "Escape") {
			e.preventDefault();
			triggerUnseal(window.innerWidth * 0.5, window.innerHeight * 0.5);
		}
	};
	window.addEventListener("keydown", onKeyDown);

	let wheelAccum = 0;
	let wheelTimer: ReturnType<typeof setTimeout> | undefined;
	const onWheel = (e: WheelEvent) => {
		if (phase === "shattering" || phase === "done") return;
		e.preventDefault();
		if (e.deltaY > 0) {
			wheelAccum += e.deltaY;
			clearTimeout(wheelTimer);
			wheelTimer = setTimeout(() => {
				wheelAccum = 0;
			}, 500);
			if (wheelAccum >= 36) {
				triggerUnseal(window.innerWidth * 0.5, window.innerHeight * 0.5);
			}
		}
	};
	root.addEventListener("wheel", onWheel, { passive: false });

	let touchStartY = 0;
	const onTouchStart = (e: TouchEvent) => {
		if (e.touches.length === 1) {
			touchStartY = e.touches[0].clientY;
		}
	};
	const onTouchEnd = (e: TouchEvent) => {
		if (e.changedTouches.length === 1 && touchStartY > 0) {
			const dy = touchStartY - e.changedTouches[0].clientY;
			if (dy > 75) {
				triggerUnseal(e.changedTouches[0].clientX, e.changedTouches[0].clientY);
			}
			touchStartY = 0;
		}
	};
	root.addEventListener("touchstart", onTouchStart, { passive: true });
	root.addEventListener("touchend", onTouchEnd, { passive: true });

	// Cached bounding box for magnetic CTA — zero reflow on pointermove!
	let magnetRect: { cx: number; cy: number; hw: number; hh: number } | null =
		null;
	let magnetActive = false;
	const refreshMagnetRect = () => {
		if (!magneticBtn) return;
		const r = magneticBtn.getBoundingClientRect();
		if (r.width > 0 && r.height > 0) {
			magnetRect = {
				cx: r.left + r.width * 0.5,
				cy: r.top + r.height * 0.5,
				hw: r.width * 0.5 + 56,
				hh: r.height * 0.5 + 44,
			};
		}
	};

	const onMagnetMove = (e: PointerEvent) => {
		if (!magneticBtn || isCoarsePointer || reduceMotion || phase !== "idle") {
			return;
		}
		if (!magnetRect) refreshMagnetRect();
		if (!magnetRect) return;
		const dx = e.clientX - magnetRect.cx;
		const dy = e.clientY - magnetRect.cy;
		if (Math.abs(dx) < magnetRect.hw && Math.abs(dy) < magnetRect.hh) {
			magnetActive = true;
			magneticBtn.style.setProperty("--mx", `${(dx / 3.8).toFixed(1)}px`);
			magneticBtn.style.setProperty("--my", `${(dy / 3.8).toFixed(1)}px`);
		} else if (magnetActive) {
			magnetActive = false;
			magneticBtn.style.setProperty("--mx", "0px");
			magneticBtn.style.setProperty("--my", "0px");
		}
	};
	if (!isCoarsePointer && !reduceMotion) {
		root.addEventListener("pointermove", onMagnetMove, { passive: true });
	}

	// ── Three.js WebGL Physical Iridescent Glass + Caustic + 3D Prism Engine ──
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let renderer: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let scene: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let camera: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let slabGroup: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let pmremTarget: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let panelTex: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let navTex: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let causticMesh: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let causticGeo: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let causticMat: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let vortexMesh: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let vortexGeo: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let vortexMat: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js ShaderMaterial uniforms
	let vortexUniforms: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let shadowMat: any = null;

	// 3D Gyroscopic Crystal Astrolabe Sculpture objects
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let sculptureGroup: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let prismCore: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	let prismCage: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime objects
	const gyroRings: any[] = [];
	// biome-ignore lint/suspicious/noExplicitAny: three.js disposables
	const sculptureDisposables: any[] = [];
	let sculptureBaseX = 0;
	let sculptureBaseY = 0;
	let sculptureScale = 1;
	let prismSpinImpulse = 0;
	let sealHoverBlend = 0;

	const pieces: ShardPiece[] = [];
	let W = window.innerWidth;
	let H = window.innerHeight;
	let isPortrait = W < 768 || H > W * 1.15;
	let blockW = 0;
	let blockH = 0;
	let navH = 0;
	let panelH = 0;
	let navCenterY = 0;
	let panelCenterY = 0;
	let gap = 14;
	let scaleS = 1;
	let extrudeDepth = 12;
	let bevelSize = 2.4;
	let cornerRadius = 18;
	let layoutDirty = true;

	const disposeThree = () => {
		window.removeEventListener("keydown", onKeyDown);
		window.removeEventListener("resize", onResize);
		document.removeEventListener("visibilitychange", onVisibilityChange);
		root.removeEventListener("wheel", onWheel);
		root.removeEventListener("touchstart", onTouchStart);
		root.removeEventListener("touchend", onTouchEnd);
		root.removeEventListener("pointermove", onPointerMove);
		root.removeEventListener("pointerleave", onPointerLeave);
		root.removeEventListener("pointermove", onMagnetMove);
		if (audioBtn) audioBtn.removeEventListener("click", onAudioToggle);
		for (const btn of openBtns) {
			btn.removeEventListener("click", onOpenClick);
		}
		if (sealHitBtn) {
			sealHitBtn.removeEventListener("click", onSealClick);
			sealHitBtn.removeEventListener("pointerenter", onSealEnter);
			sealHitBtn.removeEventListener("pointerleave", onSealLeave);
		}
		if (magneticBtn) {
			magneticBtn.removeEventListener("pointerenter", onCtaEnter);
			magneticBtn.removeEventListener("pointerleave", onCtaLeave);
		}

		for (const p of pieces) {
			p.bodyMesh?.geometry?.dispose?.();
			p.faceMesh?.geometry?.dispose?.();
			p.shadowMesh?.geometry?.dispose?.();
			p.bodyMat?.dispose?.();
			p.faceMat?.dispose?.();
			p.morph = null;
		}
		pieces.length = 0;
		for (const d of sculptureDisposables) {
			d?.dispose?.();
		}
		sculptureDisposables.length = 0;
		gyroRings.length = 0;
		panelTex?.dispose?.();
		navTex?.dispose?.();
		causticGeo?.dispose?.();
		causticMat?.dispose?.();
		vortexGeo?.dispose?.();
		vortexMat?.dispose?.();
		shadowMat?.dispose?.();
		pmremTarget?.dispose?.();
		renderer?.dispose?.();
		renderer = null;
	};

	if (reduceMotion || !canvas) {
		root.classList.add("is-fallback");
		return () => finishAndCleanup();
	}

	const isDark = document.documentElement.classList.contains("dark");
	const rawHue = Number.parseFloat(
		getComputedStyle(document.documentElement).getPropertyValue("--hue"),
	);
	const hue = Number.isFinite(rawHue) ? ((rawHue % 360) + 360) % 360 : 345;
	const accentHex = isDark ? hslToHex(hue, 86, 72) : hslToHex(hue, 76, 52);
	const accentAltHex = isDark
		? hslToHex((hue + 52) % 360, 78, 72)
		: hslToHex((hue + 44) % 360, 70, 62);
	const accentSoftHex = isDark
		? `hsla(${hue}, 82%, 72%, 0.42)`
		: `hsla(${hue}, 76%, 50%, 0.36)`;

	try {
		renderer = new THREE.WebGLRenderer({
			canvas,
			antialias: true,
			alpha: true,
			powerPreference: "high-performance",
		});
	} catch (_) {
		root.classList.add("is-fallback");
		return () => finishAndCleanup();
	}

	renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
	renderer.setClearColor(0x000000, 0);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = isDark ? 1.18 : 1.06;

	canvas.addEventListener("webglcontextlost", (e) => {
		e.preventDefault();
		webglReady = false;
		root.classList.add("is-fallback");
	});

	scene = new THREE.Scene();

	const pmrem = new THREE.PMREMGenerator(renderer);
	const roomEnv = new RoomEnvironment();
	pmremTarget = pmrem.fromScene(roomEnv, 0.04);
	scene.environment = pmremTarget.texture;
	scene.environmentIntensity = isDark ? 0.36 : 0.6;
	roomEnv.dispose?.();
	pmrem.dispose();

	const keyLight = new THREE.DirectionalLight(
		isDark ? 0xffebf4 : 0xffffff,
		isDark ? 1.2 : 1.38,
	);
	keyLight.position.set(-1150, 1300, 520);
	scene.add(keyLight);

	const rimLight = new THREE.DirectionalLight(
		new THREE.Color(accentHex),
		isDark ? 1.35 : 0.92,
	);
	rimLight.position.set(950, -520, 560);
	scene.add(rimLight);

	const ambient = new THREE.AmbientLight(
		isDark ? 0xc8b8cc : 0xfff8fb,
		isDark ? 0.24 : 0.62,
	);
	scene.add(ambient);

	camera = new THREE.PerspectiveCamera(30, W / H, 10, 10000);

	// ── Layer 1: Living Domain-Warped Liquid Caustic Silk Backdrop Plane ──
	const causticUniforms = {
		uTime: { value: 0 },
		uPointer: { value: new THREE.Vector2(0, 0) },
		uVelocity: { value: new THREE.Vector2(0, 0) },
		uBaseA: {
			value: new THREE.Color(isDark ? 0x0b0812 : 0xf2e7ed),
		},
		uBaseB: {
			value: new THREE.Color(isDark ? 0x160e20 : 0xe4d4de),
		},
		uAccent: { value: new THREE.Color(accentHex) },
		uAccentAlt: { value: new THREE.Color(accentAltHex) },
		uDark: { value: isDark ? 1.0 : 0.0 },
		uBurst: { value: 0.0 },
	};
	causticGeo = new THREE.PlaneGeometry(1, 1);
	causticMat = new THREE.ShaderMaterial({
		vertexShader: CAUSTIC_VERT,
		fragmentShader: CAUSTIC_FRAG,
		uniforms: causticUniforms,
		transparent: true,
		depthWrite: false,
	});
	causticMesh = new THREE.Mesh(causticGeo, causticMat);
	causticMesh.position.z = -180;
	causticMesh.frustumCulled = false;
	scene.add(causticMesh);

	// ── Layer 2: 100% GPU Vortex Bokeh & Diamond-Sparkle Particle Cloud ──
	const PARTICLE_COUNT = isCoarsePointer ? 72 : 140;
	vortexGeo = new THREE.BufferGeometry();
	const vPos = new Float32Array(PARTICLE_COUNT * 3);
	const vSeed = new Float32Array(PARTICLE_COUNT * 3);
	for (let i = 0; i < PARTICLE_COUNT; i++) {
		vPos[i * 3] = (Math.random() - 0.5) * W * 1.15;
		vPos[i * 3 + 1] = (Math.random() - 0.5) * H * 1.15;
		vPos[i * 3 + 2] = -120 + Math.random() * 90;
		vSeed[i * 3] = Math.random();
		vSeed[i * 3 + 1] = Math.random();
		vSeed[i * 3 + 2] = Math.random();
	}
	vortexGeo.setAttribute("position", new THREE.BufferAttribute(vPos, 3));
	vortexGeo.setAttribute("aSeed", new THREE.BufferAttribute(vSeed, 3));
	vortexUniforms = {
		uTime: { value: 0 },
		uDpr: { value: Math.min(window.devicePixelRatio || 1, 1.5) },
		uPtrWorld: { value: new THREE.Vector2(99999, 99999) },
		uPtrActive: { value: 0 },
		uBurst: { value: 0 },
		uBurstOrigin: { value: new THREE.Vector2(0, 0) },
		uColA: { value: new THREE.Color(accentHex) },
		uColB: { value: new THREE.Color(isDark ? 0xffeef7 : accentAltHex) },
	};
	vortexMat = new THREE.ShaderMaterial({
		vertexShader: VORTEX_VERT,
		fragmentShader: VORTEX_FRAG,
		uniforms: vortexUniforms,
		transparent: true,
		depthWrite: false,
		blending: isDark ? THREE.AdditiveBlending : THREE.NormalBlending,
	});
	vortexMesh = new THREE.Points(vortexGeo, vortexMat);
	vortexMesh.frustumCulled = false;
	scene.add(vortexMesh);

	// ── Layer 3: Sculpted 3D Crystal Slab Group ───────────────────────
	slabGroup = new THREE.Group();
	scene.add(slabGroup);

	// Shared uniforms for prismatic Fresnel dispersion + caustic sweep + cursor aura + ripple!
	const sharedGlassUniforms = {
		uTime: { value: 0 },
		uCursorWorld: { value: new THREE.Vector3(99999, 99999, 0) },
		uGlowStrength: { value: 0 },
		uGlowRadius: { value: 300 },
		uGlowColor: { value: new THREE.Color(accentHex) },
		uDispersionColor: { value: new THREE.Color(accentAltHex) },
		uRippleOrigin: { value: new THREE.Vector2(0, 0) },
		uRippleAge: { value: 99.0 },
		uDark: { value: isDark ? 1.0 : 0.0 },
	};

	shadowMat = new THREE.MeshBasicMaterial({
		color: isDark ? 0x000000 : 0x3b2234,
		transparent: true,
		opacity: isDark ? 0.48 : 0.17,
		depthWrite: false,
	});

	function createGlassMaterial() {
		const mat = new THREE.MeshPhysicalMaterial({
			color: isDark ? 0x130f1a : 0xfaf5f8,
			roughness: isDark ? 0.42 : 0.27,
			metalness: isDark ? 0.06 : 0.02,
			clearcoat: isDark ? 0.68 : 0.9,
			clearcoatRoughness: isDark ? 0.15 : 0.09,
			specularIntensity: isDark ? 0.78 : 0.88,
			iridescence: isDark ? 0.68 : 0.7,
			iridescenceIOR: 1.34,
			iridescenceThicknessRange: [240, 700],
			envMapIntensity: isDark ? 0.96 : 0.94,
		});

		mat.customProgramCacheKey = () => "eg-crystal-prism-v2";
		mat.onBeforeCompile = (shader: {
			vertexShader: string;
			fragmentShader: string;
			uniforms: Record<string, { value: unknown }>;
		}) => {
			shader.uniforms.uTime = sharedGlassUniforms.uTime;
			shader.uniforms.uCursorWorld = sharedGlassUniforms.uCursorWorld;
			shader.uniforms.uGlowStrength = sharedGlassUniforms.uGlowStrength;
			shader.uniforms.uGlowRadius = sharedGlassUniforms.uGlowRadius;
			shader.uniforms.uGlowColor = sharedGlassUniforms.uGlowColor;
			shader.uniforms.uDispersionColor = sharedGlassUniforms.uDispersionColor;
			shader.uniforms.uRippleOrigin = sharedGlassUniforms.uRippleOrigin;
			shader.uniforms.uRippleAge = sharedGlassUniforms.uRippleAge;
			shader.uniforms.uDark = sharedGlassUniforms.uDark;

			shader.vertexShader = shader.vertexShader
				.replace(
					"#include <common>",
					`#include <common>
varying vec3 vEgWorldPos;`,
				)
				.replace(
					"#include <worldpos_vertex>",
					`#include <worldpos_vertex>
vEgWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;`,
				);

			shader.fragmentShader = shader.fragmentShader
				.replace(
					"#include <common>",
					`#include <common>
uniform float uTime;
uniform vec3 uCursorWorld;
uniform float uGlowStrength;
uniform float uGlowRadius;
uniform vec3 uGlowColor;
uniform vec3 uDispersionColor;
uniform vec2 uRippleOrigin;
uniform float uRippleAge;
uniform float uDark;
varying vec3 vEgWorldPos;`,
				)
				.replace(
					"#include <emissivemap_fragment>",
					`#include <emissivemap_fragment>
// 1. Cursor-tracked sakura caustic emission glow
float egD = length(vEgWorldPos.xy - uCursorWorld.xy);
float egG = uGlowStrength * exp(-1.6 * pow(egD / max(uGlowRadius, 1.0), 2.0));
totalEmissiveRadiance += uGlowColor * egG;

// 2. True Fresnel prismatic edge dispersion along 3D crystal bevel rims
vec3 egV = normalize(vViewPosition);
float egFresnel = pow(1.0 - clamp(abs(dot(normal, egV)), 0.0, 1.0), 2.6);
vec3 egPrism = mix(uGlowColor, uDispersionColor, 0.5 + 0.5 * sin(vEgWorldPos.x * 0.012 - vEgWorldPos.y * 0.01 + uTime * 1.1));
totalEmissiveRadiance += egPrism * egFresnel * mix(0.28, 0.52, uDark);

// 3. Slow gallery specular caustic beam sweeping diagonally across bevel edges
float egSweepPos = (vEgWorldPos.x * 0.65 + vEgWorldPos.y * 0.75) * 0.0022 - uTime * 0.36;
float egSweep = pow(max(0.0, cos(egSweepPos * 3.14159)), 14.0) * (0.25 + egFresnel * 2.2);
totalEmissiveRadiance += mix(vec3(1.0), uGlowColor, 0.35) * egSweep * mix(0.09, 0.16, uDark);

// 4. Traveling shockwave caustic ripple across all shards on interaction/unseal
if (uRippleAge < 1.4) {
	float egR = length(vEgWorldPos.xy - uRippleOrigin);
	float egWaveFront = uRippleAge * 1350.0;
	float egRing = exp(-pow((egR - egWaveFront) / 55.0, 2.0)) * (1.0 - uRippleAge / 1.4);
	totalEmissiveRadiance += mix(vec3(1.0), uGlowColor, 0.5) * egRing * mix(0.32, 0.65, uDark);
}`,
				);
		};
		return mat;
	}

	// ── Layer 4: Real 3D Floating Gyroscopic Crystal Astrolabe Sculpture ──
	function buildGyroscopicSculpture() {
		sculptureGroup = new THREE.Group();
		slabGroup.add(sculptureGroup);

		// Sculpted faceted optical crystal octahedron core
		const coreGeo = new THREE.OctahedronGeometry(34, 1);
		const coreMat = new THREE.MeshPhysicalMaterial({
			color: isDark ? 0x241730 : 0xfff5fa,
			roughness: 0.12,
			metalness: 0.18,
			clearcoat: 1.0,
			clearcoatRoughness: 0.05,
			iridescence: 1.0,
			iridescenceIOR: 1.45,
			iridescenceThicknessRange: [200, 800],
			envMapIntensity: isDark ? 1.8 : 1.5,
			flatShading: true,
		});
		prismCore = new THREE.Mesh(coreGeo, coreMat);
		sculptureGroup.add(prismCore);

		// Precision metallic crystal facet wireframe cage
		const cageEdges = new THREE.EdgesGeometry(coreGeo);
		const cageMat = new THREE.LineBasicMaterial({
			color: new THREE.Color(accentHex),
			transparent: true,
			opacity: isDark ? 0.72 : 0.58,
		});
		prismCage = new THREE.LineSegments(cageEdges, cageMat);
		prismCage.scale.setScalar(1.03);
		sculptureGroup.add(prismCage);

		// 3 Counter-Rotating Precision Brushed-Metallic Armillary Rings
		const ringConfigs = [
			{
				r: 56,
				tube: 1.45,
				color: accentHex,
				tiltX: 0.65,
				tiltY: 0.3,
				speedX: 0.42,
				speedY: 0.65,
				speedZ: 0.18,
			},
			{
				r: 72,
				tube: 1.15,
				color: isDark ? "#f2e8f0" : "#3a243d",
				tiltX: -0.52,
				tiltY: 0.85,
				speedX: -0.34,
				speedY: 0.48,
				speedZ: -0.25,
			},
			{
				r: 88,
				tube: 0.9,
				color: accentAltHex,
				tiltX: 1.1,
				tiltY: -0.45,
				speedX: 0.24,
				speedY: -0.36,
				speedZ: 0.32,
			},
		];

		const nodeGeo = new THREE.SphereGeometry(2.8, 10, 10);
		const nodeMat = new THREE.MeshBasicMaterial({
			color: new THREE.Color(accentHex),
		});
		sculptureDisposables.push(
			coreGeo,
			coreMat,
			cageEdges,
			cageMat,
			nodeGeo,
			nodeMat,
		);

		for (const cfg of ringConfigs) {
			const rGeo = new THREE.TorusGeometry(cfg.r, cfg.tube, 10, 56);
			const rMat = new THREE.MeshPhysicalMaterial({
				color: new THREE.Color(cfg.color),
				roughness: 0.2,
				metalness: 0.82,
				clearcoat: 0.85,
				iridescence: 0.65,
				envMapIntensity: 1.4,
			});
			const ringMesh = new THREE.Mesh(rGeo, rMat);
			ringMesh.rotation.set(cfg.tiltX, cfg.tiltY, 0);
			ringMesh.userData = {
				speedX: cfg.speedX,
				speedY: cfg.speedY,
				speedZ: cfg.speedZ,
				baseTiltX: cfg.tiltX,
				baseTiltY: cfg.tiltY,
			};

			// Tiny glowing orbital node riding each armillary ring
			const sat = new THREE.Mesh(nodeGeo, nodeMat);
			sat.position.set(cfg.r, 0, 0);
			ringMesh.add(sat);

			sculptureGroup.add(ringMesh);
			gyroRings.push(ringMesh);
			sculptureDisposables.push(rGeo, rMat);
		}
	}

	buildGyroscopicSculpture();

	// ── Cut Pattern Definitions & Pre-Allocated Buffers (0 GC/frame) ──
	const landscapeNodes: Record<string, DriftSpec> = {
		tAB: {
			bx: 0.34,
			by: 0,
			ax: 0.02,
			ay: 0,
			fx: 0.34,
			fy: 0,
			px: 0.3,
			py: 0,
			lockY: true,
		},
		tBC: {
			bx: 0.61,
			by: 0,
			ax: 0.022,
			ay: 0,
			fx: 0.27,
			fy: 0,
			px: 1.7,
			py: 0,
			lockY: true,
		},
		tCE: {
			bx: 0.84,
			by: 0,
			ax: 0.02,
			ay: 0,
			fx: 0.31,
			fy: 0,
			px: 3.1,
			py: 0,
			lockY: true,
		},
		bAD: {
			bx: 0.54,
			by: 1,
			ax: 0.018,
			ay: 0,
			fx: 0.29,
			fy: 0,
			px: 2.2,
			py: 0,
			lockY: true,
		},
		bDE: {
			bx: 0.85,
			by: 1,
			ax: 0.02,
			ay: 0,
			fx: 0.36,
			fy: 0,
			px: 4.5,
			py: 0,
			lockY: true,
		},
		J0: {
			bx: 0.54,
			by: 0.6,
			ax: 0.018,
			ay: 0.026,
			fx: 0.28,
			fy: 0.23,
			px: 0.8,
			py: 1.4,
		},
		J1: {
			bx: 0.63,
			by: 0.52,
			ax: 0.022,
			ay: 0.03,
			fx: 0.25,
			fy: 0.31,
			px: 2.5,
			py: 0.5,
		},
		J2: {
			bx: 0.76,
			by: 0.43,
			ax: 0.02,
			ay: 0.03,
			fx: 0.33,
			fy: 0.26,
			px: 4.0,
			py: 2.9,
		},
	};

	const portraitNodes: Record<string, DriftSpec> = {
		l0: {
			bx: 0,
			by: 0.61,
			ax: 0,
			ay: 0.014,
			fx: 0,
			fy: 0.31,
			px: 0,
			py: 0.5,
			lockX: true,
		},
		l1: {
			bx: 0,
			by: 0.84,
			ax: 0,
			ay: 0.014,
			fx: 0,
			fy: 0.27,
			px: 0,
			py: 2.1,
			lockX: true,
		},
		r0: {
			bx: 1,
			by: 0.58,
			ax: 0,
			ay: 0.014,
			fx: 0,
			fy: 0.29,
			px: 0,
			py: 1.4,
			lockX: true,
		},
		r1: {
			bx: 1,
			by: 0.82,
			ax: 0,
			ay: 0.014,
			fx: 0,
			fy: 0.34,
			px: 0,
			py: 3.3,
			lockX: true,
		},
		b0: {
			bx: 0.48,
			by: 1,
			ax: 0.02,
			ay: 0,
			fx: 0.28,
			fy: 0,
			px: 1.9,
			py: 0,
			lockY: true,
		},
		J0: {
			bx: 0.36,
			by: 0.62,
			ax: 0.018,
			ay: 0.016,
			fx: 0.26,
			fy: 0.32,
			px: 0.6,
			py: 2.0,
		},
		J1: {
			bx: 0.66,
			by: 0.59,
			ax: 0.018,
			ay: 0.016,
			fx: 0.31,
			fy: 0.24,
			px: 2.4,
			py: 0.9,
		},
		J2: {
			bx: 0.5,
			by: 0.8,
			ax: 0.02,
			ay: 0.016,
			fx: 0.29,
			fy: 0.28,
			px: 3.9,
			py: 1.5,
		},
	};

	const LANDSCAPE_DEFS: string[][] = [
		["TL", "tAB", "J0", "bAD", "BL"],
		["tAB", "tBC", "J1", "J0"],
		["tBC", "tCE", "J2", "J1"],
		["J0", "J1", "J2", "bDE", "bAD"],
		["tCE", "TR", "BR", "bDE", "J2"],
	];

	const PORTRAIT_DEFS: string[][] = [
		["TL", "TR", "r0", "J1", "J0", "l0"],
		["l0", "J0", "J2", "l1"],
		["J0", "J1", "r0", "r1", "J2"],
		["l1", "J2", "b0", "BL"],
		["J2", "r1", "BR", "b0"],
	];

	const scratchNodes: Record<string, Vec2> = {};
	let activeNodeEntries: Array<[string, DriftSpec]> = [];
	let curPolys: Vec2[][] = [];

	function resetPolygonBuffers() {
		const spec = isPortrait ? portraitNodes : landscapeNodes;
		const defs = isPortrait ? PORTRAIT_DEFS : LANDSCAPE_DEFS;
		activeNodeEntries = Object.entries(spec);
		scratchNodes.TL = { x: -blockW * 0.5, y: panelH * 0.5 };
		scratchNodes.TR = { x: blockW * 0.5, y: panelH * 0.5 };
		scratchNodes.BR = { x: blockW * 0.5, y: -panelH * 0.5 };
		scratchNodes.BL = { x: -blockW * 0.5, y: -panelH * 0.5 };
		for (const [k] of activeNodeEntries) {
			scratchNodes[k] = { x: 0, y: 0 };
		}
		curPolys = defs.map((names) => names.map((n) => scratchNodes[n]));
	}

	function evaluatePolygonsInPlace(
		timeSec: number,
		ptrU: number,
		ptrV: number,
		ptrActive: number,
	) {
		const halfW = blockW * 0.5;
		const halfH = panelH * 0.5;
		for (let i = 0; i < activeNodeEntries.length; i++) {
			const [k, d] = activeNodeEntries[i];
			let ux = d.bx + Math.sin(timeSec * d.fx + d.px) * d.ax;
			let uy = d.by + Math.sin(timeSec * d.fy + d.py) * d.ay;
			if (ptrActive > 0.01) {
				const dx = ux - ptrU;
				const dy = uy - ptrV;
				const dist = Math.hypot(dx, dy);
				if (dist > 0.001 && dist < 0.58) {
					const push = (1 - dist / 0.58) * 0.042 * ptrActive;
					if (!d.lockX) ux += (dx / dist) * push;
					if (!d.lockY) uy += (dy / dist) * push;
				}
			}
			if (!d.lockX) ux = clamp(ux, 0.14, 0.88);
			if (!d.lockY) uy = clamp(uy, 0.14, 0.88);
			const pt = scratchNodes[k];
			pt.x = ux * blockW - halfW;
			pt.y = halfH - uy * panelH;
		}
	}

	function buildShardOnce(piece: ShardPiece, restWorldPoly: Vec2[]) {
		const restCopy = restWorldPoly.map((p) => ({ x: p.x, y: p.y }));
		const insetPts = insetPolygon(restCopy, gap * 0.5);
		const centroid = polygonCentroid(insetPts);
		const localPts = insetPts.map((p) => ({
			x: p.x - centroid.x,
			y: p.y - centroid.y,
		}));

		const shape = buildRoundedShape(localPts, cornerRadius);
		const bodyGeo = new THREE.ExtrudeGeometry(shape, {
			depth: extrudeDepth,
			bevelEnabled: true,
			bevelThickness: bevelSize,
			bevelSize,
			bevelSegments: 1,
			curveSegments: 3,
		});

		const faceGeo = new THREE.ShapeGeometry(shape, 3);
		const facePosAttr = faceGeo.attributes.position;
		const uvs = new Float32Array(facePosAttr.count * 2);
		for (let i = 0; i < facePosAttr.count; i++) {
			const wx = facePosAttr.getX(i) + centroid.x;
			const wy = facePosAttr.getY(i) + centroid.y;
			uvs[i * 2] = wx / blockW + 0.5;
			uvs[i * 2 + 1] = wy / panelH + 0.5;
		}
		const uvAttr = new THREE.BufferAttribute(uvs, 2);
		uvAttr.setUsage(THREE.DynamicDrawUsage);
		faceGeo.setAttribute("uv", uvAttr);

		const shadowGeo = new THREE.ShapeGeometry(shape, 2);

		const centroidWeights = computeMVC(centroid.x, centroid.y, restCopy);
		const bodySource = new Uint32Array(bodyGeo.attributes.position.count);
		const bodyWeights = bakeMorphWeights(
			bodyGeo,
			centroid.x,
			centroid.y,
			restCopy,
			bodySource,
		);
		const faceWeights = bakeMorphWeights(
			faceGeo,
			centroid.x,
			centroid.y,
			restCopy,
		);
		const shadowWeights = bakeMorphWeights(
			shadowGeo,
			centroid.x,
			centroid.y,
			restCopy,
		);

		piece.bodyMesh.geometry?.dispose?.();
		piece.faceMesh.geometry?.dispose?.();
		piece.shadowMesh.geometry?.dispose?.();

		piece.bodyMesh.geometry = bodyGeo;
		piece.faceMesh.geometry = faceGeo;
		piece.shadowMesh.geometry = shadowGeo;

		piece.faceMesh.position.z = extrudeDepth + bevelSize + 0.45;
		piece.shadowMesh.position.z = -10;
		piece.baseX = centroid.x;
		piece.baseY = centroid.y + panelCenterY;

		piece.morph = {
			k: restCopy.length,
			centroidWeights,
			bodyWeights,
			bodySource,
			faceWeights,
			shadowWeights,
		};
	}

	function morphShardInPlace(piece: ShardPiece, poly: Vec2[]) {
		const m = piece.morph;
		if (!m) return;
		const k = m.k;
		const cw = m.centroidWeights;

		let cx = 0;
		let cy = 0;
		for (let j = 0; j < k; j++) {
			const w = cw[j];
			cx += w * poly[j].x;
			cy += w * poly[j].y;
		}
		piece.baseX = cx;
		piece.baseY = cy + panelCenterY;

		const bodyPos = piece.bodyMesh.geometry.attributes.position;
		const bArr = bodyPos.array as Float32Array;
		const bW = m.bodyWeights;
		const bCount: number = bodyPos.count;
		for (let v = 0, pi = 0; v < bCount; v++, pi += 3) {
			const source = m.bodySource[v] * 3;
			if (source !== pi) {
				bArr[pi] = bArr[source];
				bArr[pi + 1] = bArr[source + 1];
				continue;
			}
			let wx = 0;
			let wy = 0;
			for (let j = 0; j < k; j++) {
				const w = bW[v * k + j];
				wx += w * poly[j].x;
				wy += w * poly[j].y;
			}
			bArr[pi] = wx - cx;
			bArr[pi + 1] = wy - cy;
		}
		bodyPos.needsUpdate = true;

		const faceGeo = piece.faceMesh.geometry;
		const facePos = faceGeo.attributes.position;
		const faceUv = faceGeo.attributes.uv;
		const fArr = facePos.array as Float32Array;
		const uvArr = faceUv.array as Float32Array;
		const fW = m.faceWeights;
		const fCount: number = facePos.count;
		const invBlockW = 1 / blockW;
		const invPanelH = 1 / panelH;
		for (let v = 0, wi = 0, pi = 0, ui = 0; v < fCount; v++, pi += 3, ui += 2) {
			let wx = 0;
			let wy = 0;
			for (let j = 0; j < k; j++, wi++) {
				const w = fW[wi];
				wx += w * poly[j].x;
				wy += w * poly[j].y;
			}
			fArr[pi] = wx - cx;
			fArr[pi + 1] = wy - cy;
			uvArr[ui] = wx * invBlockW + 0.5;
			uvArr[ui + 1] = wy * invPanelH + 0.5;
		}
		facePos.needsUpdate = true;
		faceUv.needsUpdate = true;

		const shPos = piece.shadowMesh.geometry.attributes.position;
		const sArr = shPos.array as Float32Array;
		const sW = m.shadowWeights;
		const sCount: number = shPos.count;
		for (let v = 0, wi = 0, pi = 0; v < sCount; v++, pi += 3) {
			let wx = 0;
			let wy = 0;
			for (let j = 0; j < k; j++, wi++) {
				const w = sW[wi];
				wx += w * poly[j].x;
				wy += w * poly[j].y;
			}
			sArr[pi] = wx - cx;
			sArr[pi + 1] = wy - cy;
		}
		shPos.needsUpdate = true;
	}

	function buildNavOnce(piece: ShardPiece) {
		const shape = buildRoundedRectShape(blockW, navH, cornerRadius);
		const bodyGeo = new THREE.ExtrudeGeometry(shape, {
			depth: extrudeDepth * 0.9,
			bevelEnabled: true,
			bevelThickness: bevelSize * 0.9,
			bevelSize: bevelSize * 0.9,
			bevelSegments: 1,
			curveSegments: 3,
		});

		const faceGeo = new THREE.ShapeGeometry(shape, 3);
		const posAttr = faceGeo.attributes.position;
		const uvs = new Float32Array(posAttr.count * 2);
		for (let i = 0; i < posAttr.count; i++) {
			uvs[i * 2] = posAttr.getX(i) / blockW + 0.5;
			uvs[i * 2 + 1] = posAttr.getY(i) / navH + 0.5;
		}
		faceGeo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));

		const shadowGeo = new THREE.ShapeGeometry(shape, 2);

		piece.bodyMesh.geometry?.dispose?.();
		piece.faceMesh.geometry?.dispose?.();
		piece.shadowMesh.geometry?.dispose?.();

		piece.bodyMesh.geometry = bodyGeo;
		piece.faceMesh.geometry = faceGeo;
		piece.shadowMesh.geometry = shadowGeo;

		piece.faceMesh.position.z = extrudeDepth * 0.9 + bevelSize * 0.9 + 0.45;
		piece.shadowMesh.position.z = -10;
		piece.baseX = 0;
		piece.baseY = navCenterY;
	}

	function createPieces() {
		const ids = ["nav", "A", "B", "C", "D", "E"];
		const dirs: Array<[number, number, number]> = [
			[0, 320, 560],
			[-420, 120, 680],
			[-80, 360, 740],
			[220, 310, 790],
			[-60, -350, 620],
			[480, -140, 760],
		];

		for (let i = 0; i < ids.length; i++) {
			const isNav = i === 0;
			const group = new THREE.Group();
			const bodyMat = createGlassMaterial();
			const faceMat = new THREE.MeshBasicMaterial({
				map: isNav ? navTex : panelTex,
				transparent: true,
				depthWrite: false,
				toneMapped: false,
			});

			const bodyMesh = new THREE.Mesh(new THREE.BufferGeometry(), bodyMat);
			const faceMesh = new THREE.Mesh(new THREE.BufferGeometry(), faceMat);
			const shadowMesh = new THREE.Mesh(new THREE.BufferGeometry(), shadowMat);
			bodyMesh.frustumCulled = false;
			faceMesh.frustumCulled = false;
			shadowMesh.frustumCulled = false;

			group.add(shadowMesh);
			group.add(bodyMesh);
			group.add(faceMesh);
			slabGroup.add(group);

			const axis = new THREE.Vector3(
				Math.random() * 2 - 1,
				Math.random() * 2 - 1,
				(Math.random() * 2 - 1) * 0.6,
			).normalize();

			pieces.push({
				id: ids[i],
				isNav,
				group,
				bodyMesh,
				faceMesh,
				shadowMesh,
				bodyMat,
				faceMat,
				morph: null,
				baseX: 0,
				baseY: 0,
				lift: 0,
				liftVel: 0,
				tipX: 0,
				tipY: 0,
				tipXVel: 0,
				tipYVel: 0,
				targetTipX: 0,
				targetTipY: 0,
				hoverBlend: 0,
				flyDelay: i === 0 ? 0 : 0.03 + i * 0.028,
				flyDirX: dirs[i][0],
				flyDirY: dirs[i][1],
				flyDirZ: dirs[i][2],
				tumbleAxis: axis,
				tumbleAngle: 2.1 + Math.random() * 1.4,
			});
		}
	}

	function rebuildLayoutAndTextures() {
		W = window.innerWidth;
		H = window.innerHeight;
		isPortrait = W < 768 || H > W * 1.15;

		const dpr = Math.min(window.devicePixelRatio || 1, 1.25);
		renderer.setPixelRatio(dpr);
		renderer.setSize(W, H, false);
		if (vortexUniforms) vortexUniforms.uDpr.value = dpr;

		const fovRad = (30 * Math.PI) / 180;
		const camZ = H / (2 * Math.tan(fovRad / 2));
		camera.aspect = W / H;
		camera.near = Math.max(10, camZ * 0.04);
		camera.far = camZ * 4.5;
		camera.position.set(0, 0, camZ);
		camera.lookAt(0, 0, 0);
		camera.updateProjectionMatrix();

		// Scale backdrop caustic plane to cover frustum cleanly at z = -180
		const bgScale = (camZ + 180) / camZ;
		causticMesh.scale.set(W * bgScale * 1.04, H * bgScale * 1.04, 1);

		if (isPortrait) {
			blockW = Math.max(280, W - 32);
			blockH = Math.max(460, H - 36);
			scaleS = clamp(Math.min(blockW, blockH) / 900, 0.48, 0.9);
			gap = Math.max(8, 11 * scaleS);
			navH = Math.max(46, Math.round(blockH * 0.085));
		} else {
			const marginX = W * 0.055;
			const marginY = H * 0.08;
			blockW = clamp(W - marginX * 2, 640, 1480);
			blockH = clamp(H - marginY * 2, 460, 860);
			scaleS = clamp(blockW / 1600, 0.52, 1.05);
			gap = Math.max(10, 14 * scaleS);
			navH = Math.max(52, Math.round(blockH * 0.105));
		}

		panelH = Math.max(240, blockH - navH - gap);
		navCenterY = blockH * 0.5 - navH * 0.5;
		panelCenterY = -blockH * 0.5 + panelH * 0.5;

		extrudeDepth = 13 * scaleS;
		bevelSize = 2.5 * scaleS;
		cornerRadius = 19 * scaleS;
		sharedGlassUniforms.uGlowRadius.value = 310 * scaleS;
		planeZ.constant = -(extrudeDepth + bevelSize);

		const texDpr = clamp(window.devicePixelRatio || 1.25, 1.25, 1.5);
		const navCanvas = renderNavCanvas(
			blockW,
			navH,
			texDpr,
			isDark,
			accentHex,
			meta,
			isPortrait,
		);
		const panelResult = renderPanelCanvas(
			blockW,
			panelH,
			texDpr,
			isDark,
			accentHex,
			accentSoftHex,
			hue,
			meta,
			isPortrait,
		);

		navTex?.dispose?.();
		panelTex?.dispose?.();
		navTex = new THREE.CanvasTexture(navCanvas);
		navTex.colorSpace = THREE.SRGBColorSpace;
		navTex.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());

		panelTex = new THREE.CanvasTexture(panelResult.canvas);
		panelTex.colorSpace = THREE.SRGBColorSpace;
		panelTex.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());

		for (const p of pieces) {
			p.faceMat.map = p.isNav ? navTex : panelTex;
			p.faceMat.needsUpdate = true;
		}

		// Position real 3D gyroscopic crystal sculpture directly over the etched blueprint dial!
		sculptureBaseX = (panelResult.sealNormX - 0.5) * blockW;
		sculptureBaseY = (0.5 - panelResult.sealNormY) * panelH + panelCenterY;
		sculptureScale = clamp(panelResult.sealRadiusPx / 106, 0.52, 1.48);
		if (sculptureGroup) {
			sculptureGroup.position.set(
				sculptureBaseX,
				sculptureBaseY,
				extrudeDepth + bevelSize + 118 * scaleS,
			);
			sculptureGroup.scale.setScalar(sculptureScale);
		}

		if (panelOverlay) {
			panelOverlay.style.width = `${blockW}px`;
			panelOverlay.style.height = `${panelH}px`;
			panelOverlay.dataset.layout = isPortrait ? "portrait" : "landscape";
			panelOverlay.style.setProperty(
				"--eg-cta-top",
				`${panelResult.ctaTopPct.toFixed(2)}%`,
			);
			panelOverlay.style.setProperty(
				"--eg-seal-top",
				`${panelResult.sealTopPct.toFixed(2)}%`,
			);
		}
		if (navOverlay) {
			navOverlay.style.width = `${blockW}px`;
			navOverlay.style.height = `${navH}px`;
			navOverlay.dataset.wide =
				!isPortrait && blockW >= 1040 ? "true" : "false";
		}

		resetPolygonBuffers();
		evaluatePolygonsInPlace(0, 0.5, 0.5, 0);
		buildNavOnce(pieces[0]);
		for (let i = 0; i < 5; i++) {
			buildShardOnce(pieces[i + 1], curPolys[i]);
		}

		layoutDirty = true;
		magnetRect = null;
	}

	const planeZ = new THREE.Plane(new THREE.Vector3(0, 0, 1), -15);
	createPieces();
	rebuildLayoutAndTextures();

	let resizeTimer: ReturnType<typeof setTimeout> | undefined;
	const onResize = () => {
		clearTimeout(resizeTimer);
		resizeTimer = setTimeout(() => {
			if (!disposed) rebuildLayoutAndTextures();
		}, 120);
	};
	window.addEventListener("resize", onResize);

	const onVisibilityChange = () => {
		if (!document.hidden && !disposed && !rafId) {
			lastTime = performance.now();
			rafId = requestAnimationFrame(tick);
		}
	};
	document.addEventListener("visibilitychange", onVisibilityChange);

	// ── Pointer Tracking, Velocity Wake & Optical Reticle ────────────
	const pointerNdc = new THREE.Vector2(99, 99);
	const raycaster = new THREE.Raycaster();
	const worldHit = new THREE.Vector3();
	const localHit = new THREE.Vector3();
	let ptrActive = 0;
	let targetPtrActive = 0;
	let ptrNormX = 0;
	let ptrNormY = 0;
	let ptrVelX = 0;
	let ptrVelY = 0;
	let ptrClientX = W * 0.5;
	let ptrClientY = H * 0.5;
	let retX = W * 0.5;
	let retY = H * 0.5;
	let retSeeded = false;
	let ptrPanelU = 0.5;
	let ptrPanelV = 0.5;
	let smoothTiltX = 0;
	let smoothTiltY = 0;
	let lastHomoTiltX = 999;
	let lastHomoTiltY = 999;
	let hoveredIndex = -1;
	let prevHoveredIndex = -1;

	const onPointerMove = (e: PointerEvent) => {
		if (phase === "shattering" || phase === "done") return;
		const nx = (e.clientX / W) * 2 - 1;
		const ny = -(e.clientY / H) * 2 + 1;
		const dvx = nx - ptrNormX;
		const dvy = ny - ptrNormY;
		ptrVelX = clamp(ptrVelX + dvx * 2.4, -1.5, 1.5);
		ptrVelY = clamp(ptrVelY + dvy * 2.4, -1.5, 1.5);

		// Sweeping mouse near the 3D crystal sculpture adds physical spin impulse!
		const speed = Math.hypot(dvx, dvy);
		if (speed > 0.008) {
			prismSpinImpulse = Math.min(4.2, prismSpinImpulse + speed * 8.5);
		}

		pointerNdc.set(nx, ny);
		ptrNormX = nx;
		ptrNormY = ny;
		ptrClientX = e.clientX;
		ptrClientY = e.clientY;
		targetPtrActive = 1;

		if (!retSeeded && reticleEl && !isCoarsePointer) {
			retX = e.clientX;
			retY = e.clientY;
			retSeeded = true;
			reticleEl.classList.add("is-visible");
		}

		const px = (e.clientX - W * 0.5) / blockW + 0.5;
		const py = (H * 0.5 - e.clientY - panelCenterY) / panelH;
		ptrPanelU = clamp(px, 0, 1);
		ptrPanelV = clamp(0.5 - py, 0, 1);
	};
	const onPointerLeave = () => {
		targetPtrActive = 0;
		hoveredIndex = -1;
		pointerNdc.set(99, 99);
		reticleEl?.classList.remove("is-visible", "is-locked");
	};
	root.addEventListener("pointermove", onPointerMove, { passive: true });
	root.addEventListener("pointerleave", onPointerLeave, { passive: true });

	const cornerVecs = [
		new THREE.Vector3(),
		new THREE.Vector3(),
		new THREE.Vector3(),
		new THREE.Vector3(),
	];
	const dstCorners: Vec2[] = [
		{ x: 0, y: 0 },
		{ x: 0, y: 0 },
		{ x: 0, y: 0 },
		{ x: 0, y: 0 },
	];
	const tumbleQuat = new THREE.Quaternion();
	const hoverEuler = new THREE.Euler(0, 0, 0, "YXZ");

	function syncHomography(
		el: HTMLElement | null,
		w: number,
		h: number,
		cy: number,
		topZ: number,
	) {
		if (!el) return;
		const hw = w * 0.5;
		const hh = h * 0.5;
		cornerVecs[0].set(-hw, cy + hh, topZ);
		cornerVecs[1].set(hw, cy + hh, topZ);
		cornerVecs[2].set(hw, cy - hh, topZ);
		cornerVecs[3].set(-hw, cy - hh, topZ);

		for (let i = 0; i < 4; i++) {
			cornerVecs[i].applyMatrix4(slabGroup.matrixWorld);
			cornerVecs[i].project(camera);
			dstCorners[i].x = (cornerVecs[i].x * 0.5 + 0.5) * W;
			dstCorners[i].y = (-cornerVecs[i].y * 0.5 + 0.5) * H;
		}
		const matStr = solveMatrix3d(w, h, dstCorners);
		if (matStr) {
			el.style.transform = matStr;
		}
	}

	// ── Main 60/120fps Zero-Allocation Render Loop ───────────────────
	let lastTime = performance.now();
	let lastMorphTs = 0;
	const startTime = lastTime;

	renderer.compile(scene, camera);
	webglReady = true;
	root.classList.add("is-webgl-ready");
	triggerScramble(0);

	setTimeout(() => {
		if (!disposed && webglReady && fallbackEl) {
			fallbackEl.style.display = "none";
		}
		if (!disposed) refreshMagnetRect();
	}, 320);

	const tick = (now: number) => {
		if (disposed) return;
		if (document.hidden) {
			rafId = 0;
			return;
		}

		const dt = Math.min(0.05, Math.max(0.001, (now - lastTime) / 1000));
		lastTime = now;
		const elapsed = (now - startTime) / 1000;
		const phaseElapsed = (now - phaseStart) / 1000;

		// Decay cursor wake velocity & prism angular impulse smoothly
		const velDecay = Math.exp(-dt * 5.5);
		ptrVelX *= velDecay;
		ptrVelY *= velDecay;
		prismSpinImpulse *= Math.exp(-dt * 2.6);

		// Autonomous gentle shimmer sweep on touch/mobile
		if (isCoarsePointer && phase === "idle") {
			const autoX = Math.sin(elapsed * 0.55) * 0.45;
			const autoY = Math.cos(elapsed * 0.42) * 0.3;
			ptrNormX = autoX;
			ptrNormY = autoY;
			ptrPanelU = 0.5 + autoX * 0.35;
			ptrPanelV = 0.5 - autoY * 0.35;
			targetPtrActive = 0.72;
			pointerNdc.set(autoX, autoY);
		}

		const ptrLerp = 1 - Math.exp(-dt * 6);
		ptrActive += (targetPtrActive - ptrActive) * ptrLerp;

		// Update shared GPU shader uniforms (caustic backdrop + prismatic glass + vortex particles)
		const burstProgress =
			phase === "shattering" ? clamp(phaseElapsed / 0.62, 0, 1) : 0;

		causticUniforms.uTime.value = elapsed;
		causticUniforms.uPointer.value.set(ptrNormX, ptrNormY);
		causticUniforms.uVelocity.value.set(ptrVelX, ptrVelY);
		causticUniforms.uBurst.value = burstProgress;

		sharedGlassUniforms.uTime.value = elapsed;
		sharedGlassUniforms.uGlowStrength.value =
			phase === "shattering" ? 0 : ptrActive * (isDark ? 0.3 : 0.16);
		if (sharedGlassUniforms.uRippleAge.value < 1.5) {
			sharedGlassUniforms.uRippleAge.value += dt;
		}

		vortexUniforms.uTime.value = elapsed;
		vortexUniforms.uPtrActive.value = ptrActive;
		vortexUniforms.uBurst.value = burstProgress;

		// Smooth trailing precision optical reticle (pure GPU translate3d)
		if (reticleEl && retSeeded && !isCoarsePointer && phase !== "shattering") {
			const rl = 1 - Math.exp(-dt * 16);
			retX += (ptrClientX - retX) * rl;
			retY += (ptrClientY - retY) * rl;
			reticleEl.style.transform = `translate3d(${retX.toFixed(1)}px,${retY.toFixed(1)}px,0)`;
		}

		// Global 3D slab tilt toward pointer
		const targetRotY =
			phase === "shattering" ? 0 : ptrActive * (ptrNormX * 0.115 - 0.015);
		const targetRotX =
			phase === "shattering" ? 0 : ptrActive * (-ptrNormY * 0.088 + 0.012);
		const tiltLerp = 1 - Math.exp(-dt * 5.2);
		smoothTiltX += (targetRotX - smoothTiltX) * tiltLerp;
		smoothTiltY += (targetRotY - smoothTiltY) * tiltLerp;
		slabGroup.rotation.set(smoothTiltX, smoothTiltY, 0);

		if (phase === "shattering" && burstProgress > 0.4 && causticMesh.visible) {
			causticMesh.visible = false;
		}

		// Throttled 30Hz breathing Voronoi seams via zero-alloc MVC morphing (cuts CPU vertex loop in half)
		if (phase === "idle" && now - lastMorphTs >= 32) {
			lastMorphTs = now;
			evaluatePolygonsInPlace(elapsed, ptrPanelU, ptrPanelV, ptrActive);
			for (let i = 0; i < 5; i++) {
				morphShardInPlace(pieces[i + 1], curPolys[i]);
			}
		}

		// Sync 3D homography ONLY when slab tilt changes
		if (
			phase !== "shattering" &&
			(layoutDirty ||
				Math.abs(smoothTiltX - lastHomoTiltX) > 3e-5 ||
				Math.abs(smoothTiltY - lastHomoTiltY) > 3e-5)
		) {
			slabGroup.updateMatrixWorld(true);
			syncHomography(
				panelOverlay,
				blockW,
				panelH,
				panelCenterY,
				extrudeDepth + bevelSize + 1,
			);
			syncHomography(
				navOverlay,
				blockW,
				navH,
				navCenterY,
				extrudeDepth * 0.9 + bevelSize * 0.9 + 1,
			);
			lastHomoTiltX = smoothTiltX;
			lastHomoTiltY = smoothTiltY;
			layoutDirty = false;
			if (magnetRect) magnetRect = null;
		}

		// Analytic 2D point-in-polygon hit-testing (~0.002ms)
		hoveredIndex = -1;
		if (phase === "idle" && ptrActive > 0.05) {
			raycaster.setFromCamera(pointerNdc, camera);
			if (raycaster.ray.intersectPlane(planeZ, worldHit)) {
				sharedGlassUniforms.uCursorWorld.value.copy(worldHit);
				vortexUniforms.uPtrWorld.value.set(worldHit.x, worldHit.y);

				if (!isCoarsePointer) {
					slabGroup.updateMatrixWorld(true);
					localHit.copy(worldHit);
					slabGroup.worldToLocal(localHit);

					const lx = localHit.x;
					const ly = localHit.y;
					if (
						Math.abs(lx) <= blockW * 0.5 &&
						Math.abs(ly - navCenterY) <= navH * 0.5
					) {
						hoveredIndex = 0;
					} else {
						const pyPanel = ly - panelCenterY;
						for (let i = 0; i < 5; i++) {
							if (pointInPolygon(lx, pyPanel, curPolys[i])) {
								hoveredIndex = i + 1;
								break;
							}
						}
					}

					if (hoveredIndex >= 0) {
						const hp = pieces[hoveredIndex];
						const spanX = hp.isNav ? blockW * 0.5 : blockW * 0.25;
						const spanY = hp.isNav ? navH * 0.5 : panelH * 0.25;
						hp.targetTipY = clamp((lx - hp.baseX) / spanX, -1, 1) * 0.17;
						hp.targetTipX = clamp(-(ly - hp.baseY) / spanY, -1, 1) * 0.15;
					}
				}
			}
		}

		// Trigger scramble telemetry + gentle crystal chime when moving onto a new shard
		if (phase === "idle" && hoveredIndex !== prevHoveredIndex) {
			prevHoveredIndex = hoveredIndex;
			if (hoveredIndex >= 0 && !isSealHovered) {
				triggerScramble(hoveredIndex);
				if (hoveredIndex > 0) {
					audioSynth.playHoverChime(hoveredIndex);
				}
			}
		}

		if (phase === "intro" && phaseElapsed > 0.92) {
			phase = "idle";
			refreshMagnetRect();
		}

		// Update 3D Floating Gyroscopic Crystal Sculpture (core + 3 armillary rings)
		if (sculptureGroup) {
			const targetSealBlend = isSealHovered ? 1 : 0;
			sealHoverBlend +=
				(targetSealBlend - sealHoverBlend) * (1 - Math.exp(-dt * 8));

			const spinRate = 1 + prismSpinImpulse + sealHoverBlend * 1.65;
			if (prismCore && prismCage) {
				prismCore.rotation.y += dt * 0.65 * spinRate;
				prismCore.rotation.x += dt * 0.38 * spinRate;
				prismCage.rotation.copy(prismCore.rotation);
			}

			for (let r = 0; r < gyroRings.length; r++) {
				const ring = gyroRings[r];
				const ud = ring.userData;
				// When hovering the seal, rings accelerate & gently align into a unified resonance halo
				const align = sealHoverBlend * 0.45;
				ring.rotation.x +=
					dt * ud.speedX * (1 + prismSpinImpulse * 0.5 + sealHoverBlend * 1.4);
				ring.rotation.y +=
					dt * ud.speedY * (1 + prismSpinImpulse * 0.5 + sealHoverBlend * 1.4);
				ring.rotation.z += dt * ud.speedZ * (1 + align);
				const ringScale = 1 + sealHoverBlend * (0.08 + r * 0.045);
				ring.scale.setScalar(ringScale);
			}

			if (phase === "intro") {
				const rawT = clamp((phaseElapsed - 0.14) / 0.72, 0, 1);
				const e = easeOutBack(rawT);
				sculptureGroup.position.set(
					sculptureBaseX,
					sculptureBaseY,
					extrudeDepth + bevelSize + 118 * scaleS + (1 - e) * 320 * scaleS,
				);
				sculptureGroup.scale.setScalar(Math.max(0.01, sculptureScale * e));
			} else if (phase === "shattering") {
				const e = easeInCubic(burstProgress);
				sculptureGroup.position.set(
					sculptureBaseX + 220 * scaleS * e,
					sculptureBaseY + 110 * scaleS * e,
					extrudeDepth + bevelSize + 118 * scaleS + 460 * scaleS * e,
				);
				sculptureGroup.scale.setScalar(sculptureScale * (1 + e * 0.85));
			} else {
				const bobY = Math.sin(elapsed * 1.6) * 5 * scaleS;
				const liftZ = sealHoverBlend * 32 * scaleS;
				sculptureGroup.position.set(
					sculptureBaseX,
					sculptureBaseY + bobY,
					extrudeDepth + bevelSize + 118 * scaleS + liftZ,
				);
				sculptureGroup.scale.setScalar(
					sculptureScale * (1 + sealHoverBlend * 0.1),
				);
			}
		}

		// Update 6 glass shard pieces
		const K = 170;
		const C = 19;
		for (let i = 0; i < pieces.length; i++) {
			const p = pieces[i];
			const isHovered = phase === "idle" && i === hoveredIndex;

			const targetLift = isHovered ? (p.isNav ? 28 : 48) * scaleS : 0;
			const tTipX = isHovered ? p.targetTipX : 0;
			const tTipY = isHovered ? p.targetTipY : 0;

			p.liftVel += ((targetLift - p.lift) * K - p.liftVel * C) * dt;
			p.lift += p.liftVel * dt;

			p.tipXVel += ((tTipX - p.tipX) * K - p.tipXVel * C) * dt;
			p.tipX += p.tipXVel * dt;

			p.tipYVel += ((tTipY - p.tipY) * K - p.tipYVel * C) * dt;
			p.tipY += p.tipYVel * dt;

			const hbTarget = isHovered ? 1 : 0;
			p.hoverBlend += (hbTarget - p.hoverBlend) * (1 - Math.exp(-dt * 10));
			p.bodyMat.iridescence =
				(isDark ? 0.68 : 0.7) + p.hoverBlend * (isDark ? 0.32 : 0.3);
			p.bodyMat.envMapIntensity =
				(isDark ? 0.96 : 0.94) + p.hoverBlend * (isDark ? 0.95 : 0.8);

			if (phase === "intro") {
				const rawT = clamp((phaseElapsed - p.flyDelay) / 0.68, 0, 1);
				const e = easeOutBack(rawT);
				const inv = 1 - e;
				p.group.position.set(
					p.baseX + p.flyDirX * scaleS * 0.45 * inv,
					p.baseY + p.flyDirY * scaleS * 0.45 * inv,
					p.flyDirZ * scaleS * 0.55 * inv,
				);
				tumbleQuat.setFromAxisAngle(p.tumbleAxis, p.tumbleAngle * 0.35 * inv);
				p.group.quaternion.copy(tumbleQuat);
			} else if (phase === "shattering") {
				const rawT = clamp((phaseElapsed - p.flyDelay) / 0.62, 0, 1);
				const e = easeInCubic(rawT);
				const grav = -220 * scaleS * rawT * rawT;
				p.group.position.set(
					p.baseX + p.flyDirX * scaleS * 1.75 * e,
					p.baseY + p.flyDirY * scaleS * 1.55 * e + grav,
					p.lift + p.flyDirZ * scaleS * 0.68 * e,
				);
				tumbleQuat.setFromAxisAngle(p.tumbleAxis, p.tumbleAngle * e);
				p.group.quaternion.copy(tumbleQuat);
			} else {
				p.group.position.set(p.baseX, p.baseY, p.lift);
				hoverEuler.set(p.tipX, p.tipY, 0);
				p.group.quaternion.setFromEuler(hoverEuler);
			}
		}

		renderer.render(scene, camera);
		rafId = requestAnimationFrame(tick);
	};

	rafId = requestAnimationFrame(tick);

	return () => finishAndCleanup();
}
