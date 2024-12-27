# Set the current folder location to the current R script location
library(rstudioapi)
script_path <- dirname(getSourceEditorContext()$path)
setwd(script_path)
library(ggplot2)
library(ggpubr)
library(readxl)
library(ggsignif)
library(gridExtra)

# See what system fonts are available
windowsFonts()
windowsFonts(serif = windowsFont("TT Times New Roman"))
par(family = "serif")
# Box plots with significance tests
mylist <- list(list('Forest','ALF','CBF','QYF'),
               list('Grass','DLG','DXG','HBG_S01'),
               list('Crop','JZA','YCA'))
indx <- list('a','b','c','d')
plots_list <- list()
for (i in 1:3){
  figurename <- mylist[[i]][[1]]
  for (j in 1:4){
    if (length(mylist[[i]]) >= j) {
      name <- mylist[[i]][[j]]
      print(name)
      index <- indx[[j]]
      if (j == 1) {
        data <- read_excel(paste0("../Output/EvaluationIndex_", name, ".xlsx"), range = cell_cols(1:12))
      } else {
        data <- read_excel("../Output/EvaluationIndex_station.xlsx", sheet = name, range = cell_cols(1:12))
      }
      y1 = subset(data, time == "daily")$R2
      y2 = subset(data, time == "8days")$R2
      y3 = subset(data, time == "16days")$R2
      y4 = subset(data, time == "monthly")$R2
      dd = data.frame(Group = rep(c("daily","8days","16days",'monthly'),each=18),y = c(y1,y2,y3,y4))
      dd$Group <- factor(dd$Group, levels = c("daily", "8days", "16days", "monthly"))
      max_y <- max(dd$y)
      min_y <- min(dd$y)
      divide = max_y-min_y
      # Box + Scatter Chart
      p = ggboxplot(dd,x = "Group",y = "y",color = "Group",add = "jitter") 
      # Significance plotted between two
      my_comparisons = list( c("daily", "8days"), c("8days", "16days"), c("16days", "monthly"), 
                             c("daily", "16days"),c("8days", "monthly"),c("daily", "monthly"))
      p = p + stat_compare_means(comparisons = my_comparisons,
                                 label = "p.signif",
                                 method = "t.test",
                                 size =12,
                                 label.x.npc = 0.5,  # Adjust horizontal position of the label
                                 label.y = c(max_y-divide*0.18, max_y, max_y+divide*0.18,max_y+divide*0.36, max_y+divide*0.54, max_y+divide*0.72),
                                 family = "serif")
      p = p + theme(text = element_text(family = "serif",size = 22,face = "bold"),
                    axis.title = element_text(family = "serif",size = 26,face = "bold"),
                    axis.text = element_text(family = "serif",size = 22,face = "bold"),
                    legend.text = element_text(family = "serif",size = 22,face = "bold"),
                    legend.title = element_text(family = "serif",size = 22,face = "bold"),
                    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),  # Add border around the plot
                    legend.position = "none",
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    # axis.text.x = element_blank(),
                    # axis.text.y = element_blank()
                    )
      p <- p + scale_y_continuous(limits = c(min_y, max_y + divide*0.9))
      # p <- p + labs(title = paste0('(',index,')',' ',name))
      p <- p + labs(title = paste0('(', index, ')', ' ', name)) +
        theme(plot.title = element_text(size = 26, family = "serif", face = "bold"))
      plots_list[[length(plots_list) + 1]] <- p
    }
  }
  ggsave(paste0("../Figure/BoxDisgram_",figurename,'.jpg'), 
         plot = grid.arrange(grobs = plots_list, ncol = 2), device = "jpg", 
         dpi = 1000, width = 14, height = 12)
  plots_list <- list()
}

# # Individual box diagrams
# data <- read_excel('../Output/EvaluationIndex_station.xlsx', sheet = 'ALF',range = cell_cols(1:12))
# y1 = subset(data, time == "daily")$R2
# y2 = subset(data, time == "8days")$R2
# y3 = subset(data, time == "16days")$R2
# y4 = subset(data, time == "monthly")$R2
# dd = data.frame(Group = rep(c("daily","8days","16days",'monthly'),each=18),y = c(y1,y2,y3,y4))
# dd$Group <- factor(dd$Group, levels = c("daily", "8days", "16days", "monthly"))
# max_y <- max(y4)
# min_y <- min(y1)
# # Box + Scatter Chart
# p = ggboxplot(dd,x = "Group",y = "y",color = "Group",add = "jitter") 
# # Significance plotted between two
# my_comparisons = list( c("daily", "8days"), c("8days", "16days"), c("16days", "monthly"), 
#                        c("daily", "16days"),c("8days", "monthly"),c("daily", "monthly"))
# p = p + stat_compare_means(comparisons = my_comparisons,
#                            label = "p.signif",
#                            method = "t.test",
#                            size =10,
#                            label.x.npc = 0.5,  # Adjust horizontal position of the label
#                            label.y = c(max_y-0.1, max_y-0.03, max_y+0.04,max_y+0.11, max_y+0.18, max_y+0.25),
#                            family = "Times New Roman"
#                            )
# # p + geom_signif(comparisons = my_comparisons,
# #                 map_signif_level=T,family="Times New Roman",
# #                 textsize=6,test=t.test,step_increase=0.2)
# p = p + theme(text = element_text(family = "Times New Roman",size = 20),
#               axis.title = element_text(family = "Times New Roman",size = 24),
#               axis.text = element_text(family = "Times New Roman",size = 20),
#               legend.text = element_text(family = "Times New Roman",size = 20),
#               legend.title = element_text(family = "Times New Roman",size = 20),
#               panel.border = element_rect(color = "black", fill = NA, linewidth = 1),  # Add border around the plot
#               legend.position = "none",
#               axis.title.x = element_blank(),
#               axis.title.y = element_blank(),
#               # axis.text.x = element_blank(),
#               # axis.text.y = element_blank()
#               )
# p <- p + scale_y_continuous(limits = c(min_y, max_y + 0.3))
# p <- p + labs(title = "(a) ALF")
# ggsave("../Figure/BoxDisgram_ALF.jpg", plot = p, device = "jpg", dpi = 1000, width = 8, height = 6)
