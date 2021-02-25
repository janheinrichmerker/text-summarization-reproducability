%! Class = mlutalk
%! Author = Fachschaftsrat Mathematik/Informatik der Studierendenschaft der Martin-Luther-Universität Halle-Wittenberg
%! Date = 2020/04/20

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mlutalk}[2020/04/20 v1.0]

% Encoding:
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}

% Load base class (report) and basic utilities.
\LoadClass{beamer}

% Basic utilities:
\RequirePackage{babel}
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{hyperref}

% Page layout:
\setbeamersize{text margin left=0.5cm, text margin right=0.5cm}

% Font:
\RequirePackage{libertine}
\RequirePackage{eulervm} % "Euler Maths" font for maths.
\RequirePackage[ttdefault]{sourcecodepro} % "Source Code Pro" font for monospace (code).
\RequirePackage{microtype}\DisableLigatures{family=tt*} % Disable ligatures for monospace fonts.
\RequirePackage{relsize}
\renewcommand{\UrlFont}{\ttfamily\smaller}
\newcommand{\textttsmall}[1]{\texttt{\smaller #1}}
\newcommand{\query}[1]{\textttsmall{#1}}
\newcommand{\domain}[1]{\href{http://#1}{\mbox{\textttsmall{#1}}}}
\setbeamerfont{title}{series=\bfseries,parent=structure}
\usefonttheme[onlymath]{serif}

% Math:
\RequirePackage{amsmath,amsthm,amssymb,amstext,array,cancel}
\newcommand{\union}{\cup}
\newcommand{\disjunction}{\uplus}
\newcommand{\intersection}{\cap}
\newcommand{\intersect}{\cap}
\newcommand{\infinity}{\infty}
\newcommand{\corresponds}{\triangleq}
\newcommand{\C}{\mathbb{C}} \newcommand{\complexnumbers}{\C} % Complex numbers.
\newcommand{\R}{\mathbb{R}} \newcommand{\realnumbers}{\R} % Real numbers.
\newcommand{\Q}{\mathbb{Q}} \newcommand{\rationalnumbers}{\Q} % Rational numbers.
\newcommand{\Z}{\mathbb{Z}} \newcommand{\wholenumbers}{\Z} % Whole numbers.
\newcommand{\N}{\mathbb{N}} \newcommand{\naturalnumbers}{\N} % Natural numbers.
\newcommand{\B}{\mathbb{B}} \newcommand{\binarynumbers}{\B} % Binary numbers.
\renewcommand{\O}{\mathcal{O}} % asymptotic O-Notation

% Swap commands for better style.
\let\tmp\mod \let\mod\bmod \let\bmod\tmp
\let\varemptyset\emptyset \let\emptyset\varnothing
\let\tmp\epsilon \let\epsilon\varepsilon \let\varepsilon\tmp
\let\tmp\phi \let\phi\varphi \let\varphi\tmp

\newcommand{\eqtransform}{\ensuremath{\qquad\big|\,\,}} % Equivalence transformation
\newcommand{\ditto}{\textquotedbl} \newcommand{\dito}{\ditto} % "Ditto" sign
\renewcommand{\qed}{\nopagebreak\hfill\ensuremath{\square}} % End of proof: "quod erat demonstrandum"

% Bibliography & cites:
\RequirePackage{csquotes} % Anführungszeichen und direkte Zitate
\RequirePackage[
    bibstyle=authoryear,
    citestyle=alphabetic,
    % url=false,isbn=false, % Exclude URLs/ISBNs from bibliography items.
    dashed=false, % Repeat author name in bibliography.
]{biblatex}
\renewcommand{\bibfont}{\scriptsize}
\AtEveryCite{\smaller\color{fg!40!bg}} % Make cites appear smaller and slightly transparent.
\setcounter{biburlnumpenalty}{100}
\setcounter{biburlucpenalty}{100}
\setcounter{biburllcpenalty}{100}

% Standard frames:
\newcommand{\bibliographyframe}{
    \begin{frame}[allowframebreaks]{\bibname}
        \begin{thebibliography}{long-name}
            \beamertemplatebookbibitems
            \printbibliography
        \end{thebibliography}
    \end{frame}
}
\newcommand{\titleframe}{
    \begin{frame}
        \titlepage
        % Title frame
    \end{frame}
}

% Beamer behavior:
\setbeamercovered{transparent} % Show covered elements slightly transparent instead of hiding them completely.
\beamertemplatenavigationsymbolsempty % Hide navigation controls.
\setbeamertemplate{frametitle continuation}[from second][\usebeamercolor{normal text}\color{fg!30!bg}\insertcontinuationtext] % Indicator for continued slides.
\setbeamertemplate{footline}[frame number] % Show frame number.
\setbeamertemplate{page number in head/foot}[framenumber]
\RequirePackage{appendixnumberbeamer} % Restart frame numbering at appendix.
\defbeamertemplate{description item}{align left}{\insertdescriptionitem\hfill}
\setbeamertemplate{description item}[align left]

% Set simple beamer theme.
\usetheme{boxes} % Simple

% University theme:
\definecolor{MluGreen}{RGB}{0,81,51}
\setbeamercolor{structure}{fg=MluGreen}

% Pro/contra items in lists:
\RequirePackage{bm}
\newcommand{\pro}{\item[\(\bm{+}\)]} % Pro item
\newcommand{\contra}{\item[\(\bm{-}\)]} % Contra item

% "Thank you!" banner.
\newcommand{\thankyouname}{}
\addto\captionsenglish{\renewcommand{\thankyouname}{Thank you!}}
\addto\captionsgerman{\renewcommand{\thankyouname}{Vielen Dank!}}
\addto\captionsngerman{\renewcommand{\thankyouname}{Vielen Dank!}}
\newcommand{\thankyou}{\vfill\hfill\emph{\thankyouname}}