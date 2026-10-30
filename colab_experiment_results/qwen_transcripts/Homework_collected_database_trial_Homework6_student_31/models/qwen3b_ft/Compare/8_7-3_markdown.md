![Source Image](8_7-3_source.png)
 

We can simplify the circuit to the following:

\[
5 \cdot 2 \cos 2t \textcolor{red}{(2)} \quad \textcolor{red}{(5A)} \quad 10k \Omega
\]

\[
\textcolor{red}{(5A)} \quad 10k \Omega \quad \textcolor{red}{(2)} \quad \frac{1}{30} nF \quad \Rightarrow \quad 10 \cos 2t \textcolor{red}{(A)} \quad \textcolor{red}{(1)} \quad \frac{1}{30} nF
\]

\[
\textcolor{red}{(15A)} \quad \textcolor{red}{(5A)} \quad 10k \Omega
\]

KVL:

\[
-10 \cos 2t + 15 \left( \frac{1}{30} \frac{dV_c(t)}{dt} \right) + V_c(t) = 0
\]

\(\Rightarrow\) \(\frac{1}{2} \frac{dV_c(t)}{dt} + V_c(t) = 10 \cos 2t\)

\(\Rightarrow\) \(\frac{dV_c(t)}{dt} + 2V_c(t) = 20 \cos 2t\)

\(\Rightarrow\) \(V_c(t) = A \sin(2t) + B \cos(2t)\)

