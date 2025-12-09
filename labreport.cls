\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{labreport}[2024/02/17]

\LoadClassWithOptions{article}
\RequirePackage[letterpaper, margin=1in]{geometry}
\RequirePackage{setspace}
\RequirePackage{floatrow}
\RequirePackage[colorlinks=true]{hyperref}
\RequirePackage{enumitem}
\RequirePackage{calc}
\RequirePackage{amsmath}
\RequirePackage{textcomp, gensymb}
% tables
\RequirePackage{tabularx}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{caption}
% nomenclature
\RequirePackage[stdsubgroups]{nomencl}
\RequirePackage{etoolbox}
% figures
\RequirePackage{graphicx}
\graphicspath{ {images/} }
\RequirePackage{subcaption}
% biblio
\RequirePackage[nottoc,notlot,notlof,numbib]{tocbibind}
\RequirePackage[square,sort,comma,numbers]{natbib}
% appendix
\RequirePackage[title,titletoc]{appendix}
\onehalfspacing
% code
\RequirePackage{listings}
\RequirePackage{xcolor}

%% ----------------------------------------------------------------
%% Table of Contents
% \setcounter{tocdepth}{2} % section depth to display

%% ----------------------------------------------------------------
%% Page Numbering

\def\frontmatter{
    \pagenumbering{roman}
    \setcounter{page}{1}
    \renewcommand{\thesection}{\Roman{section}}
}

\def\mainmatter{
    \pagenumbering{arabic}
    \setcounter{page}{1}
    \setcounter{section}{0}
    \renewcommand{\thesection}{\arabic{section}}
}

\def\backmatter{
    \setcounter{section}{0}
    \renewcommand{\thesection}{\Alph{section}}
}

%% ----------------------------------------------------------------
%% Title Page

% subtitle
\def\stitle{}
\newcommand{\subtitle}[1]{\def\stitle{#1\\}}

% department
\def\dept{}
\newcommand{\department}[1]{\def\dept{#1\\}}

% organisation
\def\org{}
\newcommand{\organisation}[1]{\def\org{#1\\}}

\renewcommand\titlepage{
    {
            \thispagestyle{empty}
            \maketitle
            % \newpage
        }
}

% maketitle
\renewcommand\maketitle{
    {
            \begin{center}
                \vspace*{\fill} % centre text vertically
                {\Huge \bfseries \@title }\\[4ex]
                {\Large \stitle}
                \vspace{8ex}
                {\LARGE  \@author}\\[4ex]
                {\Large \dept}
                {\Large \org}
                \vspace{4ex}
                \@date\\[8ex]
                \vspace*{\fill}
            \end{center}
        }
}

%% ----------------------------------------------------------------
%% Nomenclature & Abbr.
\makenomenclature
\setlength{\nomlabelwidth}{5em}
\newenvironment{nomen}
{
    %% This code creates the groups
    % -----------------------------------------
    \renewcommand\nomgroup[1]{
        \item[\bfseries
                    \ifstrequal{##1}{A}{Latin Letters}{% %% Latin symbols first
                        \ifstrequal{##1}{G}{Greek Letters}{% %% Greek symbols
                            \ifstrequal{##1}{M}{Abbreviations}{% %% Abbreviations
                                \ifstrequal{##1}{X}{Superscripts}{% %% Superscripts
                                    \ifstrequal{##1}{Z}{Subscripts}}}}} % %% Subscripts
              ]}
    % -----------------------------------------
    \renewcommand{\nomname}{Nomenclature and abbreviations}
    \addcontentsline{toc}{section}{\protect\numberline{}Nomenclature and abbreviations}
}
{
    \printnomenclature
    \newpage
}

\newcommand{\addnomen}[3]{#3 #2\nomenclature[#1]{#2}{\MakeUppercase#3}}

%% ----------------------------------------------------------------
%% Tables
\floatsetup[table]{capposition=t}
% \renewcommand{\arraystretch}{1.5}
\renewcommand{\arraystretch}{1}
\captionsetup{labelfont=bf, tableposition=top, figureposition=bottom}
\newenvironment{tbl}[4][\textwidth]
{
    \table[h]\centering
    \caption{#3}\label{tab:#2}
    \tabularx{#1}{#4} \hline
}
{
    \endtabularx
    \endtable
}

%% ----------------------------------------------------------------
%% Figures
\floatsetup[figure]{capposition=b}
\newenvironment{img}[2]
{
    \figure[h]\centering
    \caption{#2}\label{fig:#1}%
}
{
    \endfigure
}

\newenvironment{subimg}[4][\textwidth]
{
    \subfigure[h]{#1}\centering
    \src{#4}
    \subcaption{#3}\label{fig:#2}%
}
{
    \endsubfigure
}

\newcommand{\src}[2][\textwidth]{\includegraphics[width=#1]{#2}}

%% ----------------------------------------------------------------
%% Citations
\setcitestyle{square}
\bibliographystyle{plainnat}

%% ----------------------------------------------------------------
%% Code blocks
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

% \lstset{frame=tb,
%     language=Python,
%     aboveskip=3mm,
%     belowskip=3mm,
%     showstringspaces=false,
%     columns=flexible,
%     basicstyle={\small\ttfamily},
%     numbers=none,
%     numberstyle=\tiny\color{gray},
%     keywordstyle=\color{blue},
%     commentstyle=\color{dkgreen},
%     stringstyle=\color{mauve},
%     breaklines=true,
%     breakatwhitespace=true,
%     tabsize=3
% }

\usepackage{accsupp}

\newcommand{\noncopynumber}[1]{
    \BeginAccSupp{method=escape,ActualText={}}
    #1
    \EndAccSupp{}
}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray}\noncopynumber,
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\small,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    columns=flexible
}
\lstset{style=mystyle}