library(shiny)
library(neuralnet)
library(ggplot2)
ui <- fluidPage(
  titlePanel("ANN Training in Shiny"),
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload CSV Dataset", accept = ".csv"),
      selectInput("target", "Select Target Variable", choices = NULL),
      uiOutput("predictors"),
      numericInput("hidden", "Number of Hidden Neurons (Comma-separated)", value = 3),
      actionButton("train", "Train ANN"),
      actionButton("reset", "Reset"),
      hr(),
      h4("Model Summary"),
      verbatimTextOutput("summary")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("ANN Graph", plotOutput("annPlot")), # ANN structure graph
        tabPanel("Prediction Plot", plotOutput("predictionPlot")), # Predictions vs True Values
        tabPanel("Results Table", tableOutput("results")) # Table of predictions and true values
      )
    )
  )
)
server <- function(input, output, session) {
  data <- reactiveVal()
  ann_model <- reactiveVal()
  observeEvent(input$datafile, {
    req(input$datafile)
    df <- read.csv(input$datafile$datapath)
    data(df)
    updateSelectInput(session, "target", choices = names(df))
  })
  output$predictors <- renderUI({
    req(data())
    checkboxGroupInput("predictors", "Select Predictor Variables", choices = names(data()))
  })
  observeEvent(input$train, {
    req(data(), input$target, input$predictors, input$hidden)
    
    df <- data()
    formula <- as.formula(paste(input$target, "~", paste(input$predictors, collapse = "+")))
    hidden_layers <- as.numeric(unlist(strsplit(as.character(input$hidden), ",")))
    
    tryCatch({
      ann <- neuralnet(
        formula,
        data = df,
        hidden = hidden_layers,
        linear.output = FALSE
      )
      ann_model(ann)
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  observeEvent(input$reset, {
    data(NULL)
    ann_model(NULL)
    updateSelectInput(session, "target", choices = NULL)
    output$predictors <- renderUI({})
  })
  output$summary <- renderPrint({
    req(ann_model())
    ann_model()
  })
  output$annPlot <- renderPlot({
    req(ann_model())
    plot(ann_model())
  })
  output$predictionPlot <- renderPlot({
    req(ann_model(), data())
    df <- data()
    predictors <- input$predictors
    req(!is.null(predictors))  # Ensure predictors are selected
    
    predictions <- compute(ann_model(), df[, predictors, drop = FALSE])$net.result
    results <- data.frame(True = df[[input$target]], Predicted = predictions)
    
    ggplot(results, aes(x = True, y = Predicted)) +
      geom_point(color = "blue", size = 3) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      labs(title = "True vs Predicted Values", x = "True Values", y = "Predicted Values") +
      theme_minimal()
  })
  output$results <- renderTable({
    req(ann_model(), data())
    df <- data()
    predictors <- input$predictors
    req(!is.null(predictors))
    
    predictions <- compute(ann_model(), df[, predictors, drop = FALSE])$net.result
    data.frame(True = df[[input$target]], Predicted = round(predictions, 3))
  })
}
shinyApp(ui = ui, server = server)
