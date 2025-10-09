# SQL-Analyzer

SQL Analyzer is a Spring Boot application designed to analyze and visualize SQL execution plans.

## Prerequisites

* Java 21 or higher
* Maven 3.x

## Getting Started

### Build the Project

To build the project and create a JAR file, run:

```bash
mvn clean package
```

This will generate the executable JAR in the `target` directory.

### Running the Application

#### 1. Using Maven

You can run the application directly using Spring Boot's Maven plugin:

```bash
mvn spring-boot:run
```

#### 2. Using the Executable JAR

After building, run the application as a standalone JAR:

```bash
java -jar target/sqlplan-1.0.0.jar
```

The application will start on port 8080 by default.