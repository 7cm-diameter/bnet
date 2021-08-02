BLACK <- "#3a5068"
RED <- "#e68ca3"
GREEN <- "#48cbac"
WHITE <- "#d9d9f9"

apperance <- theme(axis.text = element_text(color = "transparent"),
                   axis.title = element_text(color = "transparent"),
                   legend.text = element_text(color = "transparent"),
                   legend.title = element_text(color = "transparent"),
                   strip.text = element_text(color = "transparent"),
                   strip.background = element_rect(color = "transparent",
                                                   fill = "transparent"),
                   axis.line = element_line(color = BLACK),
                   panel.border = element_rect(color = BLACK),
                   panel.grid.major = element_line(color = WHITE),
                   panel.grid.minor = element_line(color = WHITE))

mytheme <- theme_mixin(theme_bw(), apperance)
