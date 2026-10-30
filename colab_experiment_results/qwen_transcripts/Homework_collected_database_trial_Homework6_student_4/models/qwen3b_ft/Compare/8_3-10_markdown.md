![Source Image](8_3-10_source.png)
 Source transpose left of switch:

\[
v_s = (9A)(12\Omega) = 108V
\]

Using superposition for left side of switch:

\[
v_T = 20V \left(\frac{12\Omega}{(4+12)\Omega}\right) + 108V \left(\frac{4\Omega}{(4+12)\Omega}\right) = 42V
\]

\[
R_T = \frac{1}{\frac{1}{4\Omega} + \frac{1}{12\Omega}} = 3\Omega
\]

At \( t = 0 \), inductor acts as short:

\[
R_L = \frac{1}{\frac{1}{2\Omega} + 3\Omega} = 5.25\Omega
\]

\[
i_L = \frac{42V}{5.25\Omega} = 8A
\]

Current splitting for inductor branch:

\[
i = 8A \left(\frac{3\Omega}{7.25\Omega}\right) = 2A
\]

\[
\Rightarrow v(0) = 2A \cdot 9\Omega = 18V
\]

After \( t = 0 \):

\[
v(t > 0) = 0V
\]

\[
T = \frac{L}{R_L} = \frac{0.54H}{12\Omega} = \frac{1}{24}s, \quad R_R = 9\Omega + 3\Omega = 12\Omega
\]

\[
v(t) = 18e^{-\frac{24t}{1}}V
\]

