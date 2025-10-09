package edu.rihong.sqlplan.web;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;

@ControllerAdvice
public class GlobalExceptionHandler {
    private static final Logger logger = LoggerFactory.getLogger(GlobalExceptionHandler.class);
    
    @ExceptionHandler(Exception.class)
    public ResponseEntity<?> handleAllExceptions(Exception ex) {
        logger.error("Unhandled exception", ex);
        
        // 建议不要直接暴露 ex.getMessage 到生产，但开发阶段可以
        ProblemDetail error = new ProblemDetail();
        error.setType("ERROR");
        error.setTitle("Internal Server Error");
        error.setStatus(HttpStatus.INTERNAL_SERVER_ERROR.value());
        error.setDetail(ex.getMessage());
        return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(error); // 自动转成 JSON
    }
}