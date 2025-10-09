package edu.rihong.sqlplan.web;

import lombok.Data;

@Data
public class PlanRequest {
    private String dbPath;
    private String sql;
}