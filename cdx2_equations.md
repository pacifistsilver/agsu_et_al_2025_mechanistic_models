\begin{align*}
% 2. DNA Structural Fractions 
\frac{d(d)}{dt} &= k_{unloop} d_l - k_{loop} d - k_{bind,c} c_f d + k_{unbind,c} d_l \\[8pt]
\frac{d(d_{l})}{dt} &= k_{loop} d + k_{bind,c} c_f d - k_{unloop} d_{l} - k_{unbind, c} d_{l}\\[8pt]

% 3. CDX2 Trafficking 
\frac{d(c_{f})}{dt} &= k_{t} + k_{unloop} d_{l} + k_{unbind,c} d_l - k_{deg} c_{f} - k_{bind, c} c_{f} d \\[8pt]

% 4. SOX2 Trafficking 
\frac{d(s_{f})}{dt} &= k_{t} - k_{confine} \frac{(d_{l} / \theta_C)^{n}}{1 + (d_{l} / \theta_C)^{n}} s_{f} + k_{unconfine} s_{b} - k_{deg} s_{f} \\[8pt]
\frac{d(s_{b})}{dt} &= k_{confine} \frac{(d_{l} / \theta_C)^{n}}{1 + (d_{l} / \theta_C)^{n}} s_{f} - k_{unconfine} s_{b} \\[12pt]
\frac{dm}{dt} &= \beta s_{b} - \gamma  m
 
% kt, kbindc, kubindc, kloop, kunloop, n, beta, gamma, theta_C, kbinds, kunbinds
% m, sf, sb, d, dl, cf
\end{align*}