library(dplyr)

shinyServer(function(input, output, session) {
  
  # Provide explicit colors for regions, so they don't get recoded when the
  # different series happen to be ordered differently from year to year.
  # Mid-at, MW, NE, S, SW, W
  defaultColors <- c("#0099c6","#ff9900", "#109618", "#dc3912", "#dd4477",  "#990099")
  
  series <- structure(
    lapply(defaultColors, function(color) { list(color=color) }),
    names = levels(data2$Region)
  )
  
  yearData <- reactive({
    # put the columns in the order that Google's Bubble Chart expects them (name, x, y, color, size).
    df <- data2[match(c("State", input$idpVar, input$candidate, "Region", "ElectVotes"), names(data2))]
  })

  output$chart <- reactive({
    xlim <- list(
      min = min(data2[input$idpVar]) - mean(data2[input$idpVar])/10,
      max = max(data2[input$idpVar]) + mean(data2[input$idpVar])/10
    )
    
    # Return the data and options
    list(
      data = googleDataTable(yearData()),
      options = list(
        title = sprintf("Support rate for %s", input$idpVar),
        series = series,
        hAxis = list(
          title = input$idpVar,
          viewWindow = xlim
        )
      )
    )
  })
})