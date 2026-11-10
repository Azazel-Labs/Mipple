//! Endpoint fitting math shared by the block encoders: PCA line fit and least-squares refit.
//!
//! All functions work on up to 4 channels; `dims` selects how many (3 = RGB, 4 = RGBA,
//! 1 = single channel) are considered.

/// A texel as floats in [0, 255].
pub type V4 = [f32; 4];

/// Mean and covariance of a point set.
pub fn mean_cov(pts: &[V4], dims: usize) -> (V4, [[f32; 4]; 4]) {
    let n = pts.len().max(1) as f32;
    let mut mean = [0f32; 4];
    for p in pts {
        for c in 0..dims {
            mean[c] += p[c];
        }
    }
    for m in mean.iter_mut().take(dims) {
        *m /= n;
    }
    let mut cov = [[0f32; 4]; 4];
    for p in pts {
        let mut d = [0f32; 4];
        for c in 0..dims {
            d[c] = p[c] - mean[c];
        }
        for i in 0..dims {
            for j in i..dims {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    for i in 0..dims {
        for j in i..dims {
            cov[i][j] /= n;
            cov[j][i] = cov[i][j];
        }
    }
    (mean, cov)
}

/// Principal axis via power iteration. Returns a unit vector (zero vector if the set is flat).
pub fn principal_axis(cov: &[[f32; 4]; 4], dims: usize) -> V4 {
    // Start from the row with the largest diagonal entry; it is never orthogonal to the
    // dominant eigenvector unless that entry is zero.
    let mut best = 0;
    for c in 1..dims {
        if cov[c][c] > cov[best][best] {
            best = c;
        }
    }
    let mut v = [0f32; 4];
    v[..dims].copy_from_slice(&cov[best][..dims]);
    if v.iter().all(|x| x.abs() < 1e-12) {
        let mut u = [0f32; 4];
        for x in u.iter_mut().take(dims) {
            *x = 1.0 / (dims as f32).sqrt();
        }
        return u;
    }
    for _ in 0..8 {
        let mut w = [0f32; 4];
        for i in 0..dims {
            for j in 0..dims {
                w[i] += cov[i][j] * v[j];
            }
        }
        let len = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        if len < 1e-12 {
            break;
        }
        for c in 0..dims {
            v[c] = w[c] / len;
        }
    }
    let len = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= len;
    }
    v
}

/// Endpoints at the extremes of the point set's projection onto its principal axis.
/// Returns (low end, high end), clamped to [0, 255].
pub fn pca_endpoints(pts: &[V4], dims: usize) -> (V4, V4) {
    let (mean, cov) = mean_cov(pts, dims);
    let axis = principal_axis(&cov, dims);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for p in pts {
        let mut t = 0.0;
        for c in 0..dims {
            t += (p[c] - mean[c]) * axis[c];
        }
        lo = lo.min(t);
        hi = hi.max(t);
    }
    if pts.is_empty() {
        return (mean, mean);
    }
    let mut e0 = [0f32; 4];
    let mut e1 = [0f32; 4];
    for c in 0..dims {
        e0[c] = (mean[c] + axis[c] * lo).clamp(0.0, 255.0);
        e1[c] = (mean[c] + axis[c] * hi).clamp(0.0, 255.0);
    }
    (e0, e1)
}

/// Sum of squared distances from the points to their best-fit line (n * (trace - λmax)).
/// The scalar reference the SIMD partition rankers are tested against.
pub fn line_fit_error(pts: &[V4], dims: usize) -> f32 {
    if pts.len() < 2 {
        return 0.0;
    }
    let (_, cov) = mean_cov(pts, dims);
    let axis = principal_axis(&cov, dims);
    let mut trace = 0.0;
    let mut lambda = 0.0;
    for i in 0..dims {
        trace += cov[i][i];
        for j in 0..dims {
            lambda += axis[i] * cov[i][j] * axis[j];
        }
    }
    (trace - lambda).max(0.0) * pts.len() as f32
}

/// Least-squares endpoints for fixed interpolation weights.
///
/// Minimizes Σ |(1 - tᵢ)·e0 + tᵢ·e1 - pᵢ|². Returns `None` when the system is singular
/// (for example when every texel uses the same weight).
pub fn ls_endpoints(pts: &[V4], t: &[f32], dims: usize) -> Option<(V4, V4)> {
    let (mut aa, mut ab, mut bb) = (0f32, 0f32, 0f32);
    let mut ax = [0f32; 4];
    let mut bx = [0f32; 4];
    for (p, &ti) in pts.iter().zip(t) {
        let si = 1.0 - ti;
        aa += si * si;
        ab += si * ti;
        bb += ti * ti;
        for c in 0..dims {
            ax[c] += si * p[c];
            bx[c] += ti * p[c];
        }
    }
    let det = aa * bb - ab * ab;
    if det.abs() < 1e-6 {
        return None;
    }
    let mut e0 = [0f32; 4];
    let mut e1 = [0f32; 4];
    for c in 0..dims {
        e0[c] = ((bb * ax[c] - ab * bx[c]) / det).clamp(0.0, 255.0);
        e1[c] = ((aa * bx[c] - ab * ax[c]) / det).clamp(0.0, 255.0);
    }
    Some((e0, e1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_finds_line_extremes() {
        let pts: Vec<V4> = (0..16)
            .map(|i| {
                let t = i as f32 * 8.0;
                [t, 2.0 * t, 255.0 - t, 0.0]
            })
            .collect();
        let (a, b) = pca_endpoints(&pts, 3);
        let near = |x: V4, y: V4| (0..3).all(|c| (x[c] - y[c]).abs() < 0.5);
        let (lo, hi) = (pts[0], pts[15]);
        assert!(near(a, lo) && near(b, hi) || near(a, hi) && near(b, lo));
        assert!(line_fit_error(&pts, 3) < 1e-2);
    }

    #[test]
    fn ls_recovers_exact_endpoints() {
        let e0 = [10.0, 20.0, 30.0, 0.0];
        let e1 = [200.0, 100.0, 50.0, 0.0];
        let t: Vec<f32> = (0..16).map(|i| (i % 4) as f32 / 3.0).collect();
        let pts: Vec<V4> = t
            .iter()
            .map(|&ti| {
                let mut p = [0f32; 4];
                for c in 0..3 {
                    p[c] = (1.0 - ti) * e0[c] + ti * e1[c];
                }
                p
            })
            .collect();
        let (a, b) = ls_endpoints(&pts, &t, 3).unwrap();
        for c in 0..3 {
            assert!((a[c] - e0[c]).abs() < 1e-2 && (b[c] - e1[c]).abs() < 1e-2);
        }
    }
}
