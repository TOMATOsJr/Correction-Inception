![Source Image](10_6-1_source.png)
 

\begin{align*}
V_s(t) &= 48 \cos(2500t + 45^\circ) \textcolor{red}{V} \\
\end{align*}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{image.png}
    \caption{Circuit diagram with annotations and simplified steps.}
\end{figure}

\begin{align*}
\text{Steady-state } V_o &= ? \\
\frac{1}{j25,000} &= \frac{40}{j} \\
\end{align*}

\begin{align*}
\textcolor{red}{V_1} &= 48 \angle 45^\circ \textcolor{red}{V} \\
\textcolor{red}{V_t} &= i(\theta) \cdot 20 \textcolor{red}{A} \\
\textcolor{red}{V_2} - \textcolor{red}{V_3} &= 25 \cdot i(\theta) \\
\textcolor{red}{V_3} &= \textcolor{red}{V}_t - 25 \cdot i(\theta) \\
\end{align*}

\begin{align*}
\text{RL} \textcolor{red}{\textcircled{\textcolor{red}{@}}} \textcolor{red}{V_2} &\quad \text{Out} \\
\frac{\textcolor{red}{V}_t - \textcolor{red}{V}_1}{j50} + i(\theta) - \frac{25 \cdot i(\theta)}{\frac{128}{3} - j\frac{56}{3}} &= 0 \\
\text{RL} \textcolor{red}{\textcircled{\textcolor{red}{@}}} \textcolor{red}{V_3} &\quad \text{Out} \\
\end{align*}

