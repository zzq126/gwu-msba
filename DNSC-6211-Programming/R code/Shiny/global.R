### Program description ###
  #    With the balloon chart programmed with R Shiny,
  #    we want to see and show people the relationship between the supporting rate and each independent variable more intuitively. 
  #    On the balloon chart, each balloon means each state, and the size of the balloons are the number of the electoral votes 
  #    for each state, and each color of the balloons mean the regions where each state is located. 
  #    We could see from here the relationship between the variables including more information, the electoral votes and the regions, 
  #    as well as that our regression analysis results are pretty significant.

### Google Charts Installation : must be run before starting the Shiny app
    # if (!require(devtools))
    #   install.packages("devtools", dependencies = TRUE)
    # devtools::install_github("jcheng5/googleCharts")

### File needed to run the code : 'ProjectData.csv'

data2 <- read.csv("./ProjectData.csv")

