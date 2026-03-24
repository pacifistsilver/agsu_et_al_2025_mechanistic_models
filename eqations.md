\begin{align*}
% 1. The Topological Feedback Equation
f(d_{clust}) &= \frac{(d_{clust} / \theta)^{n}}{1 + (d_{clust} / \theta)^{n}} \\[12pt]
f(s_{nuc}) &= \frac{(s_{nuc} / \theta)^{n}}{1 + (s_{nuc} / \theta)^{n}} \\[12pt]

% 2. DNA Structural Fractions (Note: d_out + d_in + d_clust = 1)
\frac{d(d_{out})}{d\tau} &= \kappa_{D,out}  d_{in} - \kappa_{D,in}  d_{out} \\[8pt]
\frac{d(d_{in})}{d\tau} &= \kappa_{D,in}  d_{out} + \kappa_{uncluster}  d_{clust} - (\kappa_{D,out} + \kappa_{cluster})  d_{in} \\[8pt]
\frac{d(d_{clust})}{d\tau} &= \kappa_{cluster} f(s_{nuc}) d_{in} - \kappa_{uncluster}d_{clust} \\[12pt]

% 3. SOX2 Trafficking
\frac{d(s_{free})}{d\tau} &= \kappa_{prod} + \kappa_{escape}  s_{nuc} - (\kappa_{condense} + \kappa_{deg, S})  s_{cyt} \\[8pt]
\frac{d(s_{hub})}{d\tau} &= \kappa_{condense}  s_{free} - \left(\beta_{max}  f(d_{clust}) + \kappa_{escape} + \kappa_{deg, S} \right)  s_{nuc} \\[12pt]

% 4. Active Complex and mRNA
\frac{d(s_{nucleated})}{d\tau} &= \beta_{max}  f(d_{clust})  s_{in} - c \\[8pt]
\frac{dm}{d\tau} &= \kappa_{deg}  (c - m)


% new equations week 23/03/26
\frac{d(s_{f})}{d\tau} &= k_{unconfine} s_{c} + k_{prod} - k_{confine} s_{f} - k_{deg} s_{f} \\[8pt]
\frac{d(s_{c})}{d\tau} &= k_{confine} s_{f} + k_{unbind} s_{b} - k_{unconfine} s_{c} - k_{bind} s_{c}(n_{total} - s_{b}) \frac{1}{1 + (mRNA / K_{A})^{n}} \\[8pt]

\frac{d(s_{b})}{d\tau} &= k_{bind} s_{c}(n_{total} - s_{b}) \frac{1}{1 + (mRNA / K_{A})^{n}} - k_{unbind} s_{b} \\[8pt]

\frac{d(mRNA)}{d\tau} &= \beta \frac{(s_{b} / K_{A})^{n}}{1 + (s_{b} / K_{A})^{n}}  - \gamma mRNA \\[8pt]
% treat nucleosomes or "binding site" as some fixed number
\end{align*}