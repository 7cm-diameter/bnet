library(tidyverse)
source("./hbtfmt/figs/plotutil.R")

E <- 100

simdata_vi <- read.csv("./data/sim_02_variable_interval.csv")
simdata_vr <- read.csv("./data/sim_02_variable_ratio.csv")
simdata_choice <- read.csv("./data/sim_02_choice.csv")
simdata_noncontigent <- read.csv("./data/sim_02_noncontingent.csv")
matfiles <- list.files("./data", full.names = T, pattern = "mat") %>%
  (function(files) {
    idx <- files %>% grep(pattern = "sim_02")
    files[idx]
})

# preprocessing for figure 3(a, b)
preprocess_3ab <- function(d) {
  dev_ratio <- d$test / d$baseline
  dev_mean <- mean(dev_ratio)
  dev_se <- sd(dev_ratio) / sqrt(nrow(d))
  deg_prop <- d$operant.degree / E
  deg_mean <- mean(deg_prop)
  deg_se <- sd(deg_prop) / sqrt(nrow(d))
  data.frame(amount.training = unique(d$amount.training),
             dev_mean = dev_mean, dev_se = dev_se,
             deg_mean = deg_mean, deg_se = deg_se)
}

plotdata_3ab_vi <- simdata_vi %>%
  split(., list(.$amount.training)) %>%
  lapply(., preprocess_3ab) %>%
  do.call(rbind, .)

plotdata_3ab_vr <- simdata_vr %>%
  split(., list(.$amount.training)) %>%
  lapply(., preprocess_3ab) %>%
  do.call(rbind, .)

# preprocessing for figure 3d
plotdata_3d <- (function() {
                  simdata_choice$cond <- "choice"
                  simdata_noncontigent$cond <- "no choice"
                  rbind(simdata_choice, simdata_noncontigent)
             })()

# fig 3(a)
ggplot() +
  geom_line(data = plotdata_3ab_vi,
            aes(x = amount.training, y = dev_mean),
            color = RED, size = 1.5) +
  geom_line(data = plotdata_3ab_vr,
            aes(x = amount.training, y = dev_mean),
            color = BLACK, size = 1.5) +
  geom_errorbar(data = plotdata_3ab_vi,
                aes(x = amount.training,
                    ymax = dev_mean + dev_se,
                    ymin = dev_mean - dev_se),
                color = RED, width = 5, size = 1.) +
  geom_errorbar(data = plotdata_3ab_vr,
                aes(x = amount.training,
                    ymax = dev_mean + dev_se,
                    ymin = dev_mean - dev_se),
                color = BLACK, width = 5, size = 1.) +
  geom_point(data = plotdata_3ab_vi,
            aes(x = amount.training, y = dev_mean),
            color = "white", size = 6) +
  geom_point(data = plotdata_3ab_vr,
            aes(x = amount.training, y = dev_mean),
            color = "white", size = 6) +
  geom_point(data = plotdata_3ab_vi,
            aes(x = amount.training, y = dev_mean),
            color = RED, size = 4) +
  geom_point(data = plotdata_3ab_vr,
            aes(x = amount.training, y = dev_mean),
            color = BLACK, size = 4) +
  xlim(20, 205) +
  ylim(0, 1) +
  mytheme(aspect.ratio = 1.)

ggsave("./hbtfmt/figs/fig-03-a.jpg", dpi = 300, width = 10.0, height = 10.0)

# fig 3(b)
ggplot() +
  geom_line(data = plotdata_3ab_vi,
            aes(x = amount.training, y = deg_mean),
            color = RED, size = 1.5) +
  geom_line(data = plotdata_3ab_vr,
            aes(x = amount.training, y = deg_mean),
            color = BLACK, size = 1.5) +
  geom_errorbar(data = plotdata_3ab_vi,
                aes(x = amount.training,
                    ymax = deg_mean + deg_se,
                    ymin = deg_mean - deg_se),
                color = RED, width = 5, size = 1.) +
  geom_errorbar(data = plotdata_3ab_vr,
                aes(x = amount.training,
                    ymax = deg_mean + deg_se,
                    ymin = deg_mean - deg_se),
                color = BLACK, width = 5, size = 1.) +
  geom_point(data = plotdata_3ab_vi,
            aes(x = amount.training, y = deg_mean),
            color = "white", size = 6) +
  geom_point(data = plotdata_3ab_vr,
            aes(x = amount.training, y = deg_mean),
            color = "white", size = 6) +
  geom_point(data = plotdata_3ab_vi,
            aes(x = amount.training, y = deg_mean),
            color = RED, size = 4) +
  geom_point(data = plotdata_3ab_vr,
            aes(x = amount.training, y = deg_mean),
            color = BLACK, size = 4) +
  xlim(20, 205) +
  ylim(0, 0.5) +
  mytheme(aspect.ratio = 1.)

ggsave("./hbtfmt/figs/fig-03-b.jpg", dpi = 300, width = 10.0, height = 10.0)

# fig 3(c)
ggplot() +
  geom_point(data = simdata_vi,
             aes(x = operant.degree / E, y = test / baseline),
             color = RED, size = 3, alpha = 0.75) +
  geom_point(data = simdata_vr,
             aes(x = operant.degree / E, y = test / baseline),
             color = BLACK, size = 3, alpha = 0.75) +
  xlim(0.1, 0.5) +
  ylim(0., 1.) +
  mytheme(aspect.ratio = 1.)

ggsave("./hbtfmt/figs/fig-03-c.jpg", dpi = 300, width = 10.0, height = 10.0)

# fig 3(d)
ggplot() +
  geom_boxplot(data = plotdata_3d,
               aes(x = cond, y = test / baseline),
               size = 0.75, width = 0.5, color = BLACK) +
  ylim(0., 1.) +
  mytheme(aspect.ratio = 1.)

ggsave("./hbtfmt/figs/fig-03-d.jpg", dpi = 300, width = 10.0, height = 10.0)

# fig 3(e)
show_graph <- function(filename, ...) {
  mat <- read.csv(filename, header = F) %>% data.matrix
  net <- network(mat, directed = F)

  p <- ggnet2(net, edge.size = 0.5, ...) +
    theme(aspect.ratio = 1.5,
          panel.border = element_blank(),
          axis.line = element_blank(),
          axis.title = element_blank())
  return(p)
}

choice_colors <- c(RED, GREEN, rep(BLACK, 48))
noncontingent_color <- c(RED, rep(BLACK, 49))

netplot <- grid.arrange(show_graph(matfiles[[1]], color = choice_colors, size = 5),
             show_graph(matfiles[[2]], color = noncontingent_color, size = 5),
             ncol = 2)

ggsave("./hbtfmt/figs/fig-03-e.jpg", plot = netplot,
       dpi = 300, width = 20.0, height = 10.0)
