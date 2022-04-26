
rm(list=ls())

library(scales)
library(lubridate)
library(ggplot2)
library(dplyr)



# setwd

setwd("/Documents and Settings/danie/Dropbox/TLDA-Dev/")

# Load data 

tl.data <- readr::read_csv("data/timeline_data_pol.csv")

status_levels <- c("US-President", "US-Congress","US-Supreme Court" ,"Canada-PM","US-Policy")
status_colors <- c("#0070C0", "#00B050","#C00000","#FFC000","#CC5500")


tl.data$date <- with(tl.data, ymd(sprintf('%04d%02d%02d', year, month, 1)))
tl.data      <- tl.data[with(tl.data, order(date)), ]


tl.data$status <- factor(tl.data$Type, levels=status_levels, ordered=TRUE)
tl.data <- tl.data %>%mutate(direction = ifelse(Rank<0,-1,1))


month_buffer <- 2

month_date_range <- seq(min(tl.data$date) - months(month_buffer), max(tl.data$date) + months(month_buffer), by='month')
month_format <- format(month_date_range, '%b')
month_tl.data <- data.frame(month_date_range, month_format)


year_date_range <- seq(min(tl.data$date) - months(month_buffer), max(tl.data$date) + months(month_buffer), by='year')
year_date_range <- as.Date(
  intersect(
    ceiling_date(year_date_range, unit="year"),
    floor_date(year_date_range, unit="year")
  ),  origin = "1970-01-01"
)
year_format <- format(year_date_range, '%Y')
year_tl.data <- data.frame(year_date_range, year_format)

text_offset <- 0.05

tl.data$month_count <- ave(tl.data$date==tl.data$date, tl.data$date, FUN=cumsum)
tl.data$text_position <- (tl.data$month_count * text_offset * tl.data$direction) + tl.data$Rank
head(tl.data)


#### PLOT ####

timeline_plot<-ggplot(tl.data,aes(x=date,y=0, col=status, label=`Most Prominent Topic`))
timeline_plot<-timeline_plot+labs(col="Topic Types")
timeline_plot<-timeline_plot+scale_color_manual(values=status_colors, labels=status_levels, drop = FALSE)
timeline_plot<-timeline_plot+theme_classic()

# Plot horizontal black line for timeline
timeline_plot<-timeline_plot+geom_hline(yintercept=0, 
                                        color = "black", size=0.3)

# Plot vertical segment lines for milestones
timeline_plot<-timeline_plot+geom_segment(data=tl.data, aes(y=Rank,yend=0,xend=date), color='grey', size=0.2)
timeline_plot<-timeline_plot+geom_vline(xintercept =  as.Date("2017-12-31"),lty=2)
timeline_plot<-timeline_plot+geom_vline(xintercept =  as.Date("2018-12-31"),lty=2)

# Plot scatter points at zero and date
timeline_plot<-timeline_plot+geom_point(aes(y=0), size=3)

# Don't show axes, appropriately position legend
timeline_plot<-timeline_plot+theme(axis.line.y=element_blank(),
                                   axis.text.y=element_blank(),
                                   axis.title.x=element_blank(),
                                   axis.title.y=element_blank(),
                                   axis.ticks.y=element_blank(),
                                   axis.text.x =element_blank(),
                                   axis.ticks.x =element_blank(),
                                   axis.line.x =element_blank(),
                                   legend.title=element_text(size=20),
                                   legend.text=element_text(size=20),
                                   legend.position = "bottom"
)

# Show text for each month
timeline_plot<-timeline_plot+geom_label(data=month_tl.data, aes(x=month_date_range,y=-0.1,label=month_format,fontface="bold"),size=4,vjust=0.5, color='black', angle=45)
# Show year text
timeline_plot<-timeline_plot+geom_label(data=year_tl.data, aes(x=year_date_range,y=-2,label=year_format, fontface="bold"),size=4,color='black')
# Show text for each milestone
timeline_plot<-timeline_plot+geom_label(aes(y=text_position,label=`Most Prominent Topic`,fontface="bold"),size=5,show.legend = FALSE)
print(timeline_plot)


pdf("plots/prom_topics_pol.pdf",width = 15,height = 7.5)
timeline_plot
dev.off()