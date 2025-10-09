package edu.rihong.sqlplan.web;

import edu.rihong.sqlplan.service.SqlPlanService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/plan")
public class PlanController {

    @Autowired
    private SqlPlanService sqlPlanService;

    // 返回 plain text 格式
    @PostMapping(value = "/text", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.TEXT_PLAIN_VALUE)
    public String getLogicalPlanText(@RequestBody PlanRequest req) throws Exception {
        String dbPath = req.getDbPath();
        String sql = req.getSql();
        // 替换反引号
        sql = sql.replace("`", "\"");
        return sqlPlanService.getLogicalPlanText(dbPath, sql);
    }

    // 返回 XML 格式
    @PostMapping(value = "/xml", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.TEXT_XML_VALUE)
    public String getLogicalPlanXml(@RequestBody PlanRequest req) throws Exception {
        String dbPath = req.getDbPath();
        String sql = req.getSql();
        // 替换反引号
        sql = sql.replace("`", "\"");
        return sqlPlanService.getLogicalPlanXml(dbPath, sql);
    }

    // 返回 json 格式
    @PostMapping(value = "/json", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public String getLogicalPlanJson(@RequestBody PlanRequest req) throws Exception {
        String dbPath = req.getDbPath();
        String sql = req.getSql();
        // 替换反引号
        sql = sql.replace("`", "\"");
        return sqlPlanService.getLogicalPlanJson(dbPath, sql);
    }
}