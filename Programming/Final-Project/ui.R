library(googleCharts)

# Use global max/min for axes so the view window stays
# constant as the user moves between years
ylim <- list(
  min = min(data2$Trump) - 2,
  max = max(data2$Trump) + 2
)

shinyUI(fluidPage(
  # This line loads the Google Charts JS library
  googleChartsInit(),
  
  # Use the Google webfont "Source Sans Pro"
  tags$link(
    href=paste0("http://fonts.googleapis.com/css?",
                "family=Source+Sans+Pro:300,600,300italic"),
    rel="stylesheet", type="text/css"),
  tags$style(type="text/css",
             "body {font-family: 'Source Sans Pro'}"
  ),
  
  h2("The election result by each factor"),
  
  googleBubbleChart("chart",
                    width="100%", height = "475px",
                    # Set the default options for this chart; they can be overridden in server.R on a per-update basis. 
                    # https://developers.google.com/chart/interactive/docs/gallery/bubblechart
                    
                    options = list(
                      fontName = "Source Sans Pro",
                      fontSize = 13,
                      # Set axis labels and ranges
                      vAxis = list(
                        title = "Support rate (%)",
                        viewWindow = ylim
                      ),
                      # The default padding is a little too spaced out
                      chartArea = list(
                        top = 50, left = 75,
                        height = "75%", width = "75%"
                      ),
                      # Allow pan/zoom
                      explorer = list(),
                      # Set bubble visual props
                      bubble = list(
                        opacity = 0.4, stroke = "none",
                        # Hide bubble label
                        textStyle = list(
                          color = "none"
                        )
                      ),
                      # Set fonts
                      titleTextStyle = list(
                        fontSize = 16
                      ),
                      tooltip = list(
                        textStyle = list(
                          fontSize = 10
                        )
                      )
                    )
  ),
  
  tags$hr(),
  
  fluidRow(
    shiny::column(4, offset = 4,
                  selectInput('candidate', label = 'Candidate', choices = names(data2)[2:3], selected = names(data2)[3]),
                  selectInput('idpVar', label = 'Independent Variable', choices = names(data2)[4:10], selected = names(data2)[4])
    )
  )         
))