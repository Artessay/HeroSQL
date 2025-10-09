package edu.rihong.sqlplan.web;

import lombok.Data;

@Data
public class ProblemDetail {
    private String type;
    private String title;
    private int status;
    private String detail;
    private String instance;
}
