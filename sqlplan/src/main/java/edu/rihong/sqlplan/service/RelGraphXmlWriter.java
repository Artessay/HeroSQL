
package edu.rihong.sqlplan.service;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.XmlOutput;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Callback for a relational expression to dump in XML format.
 */
public class RelGraphXmlWriter extends RelGraphWriter {
  //~ Instance fields --------------------------------------------------------

  private final XmlOutput xmlOutput;
  final boolean generic = true;

  //~ Constructors -----------------------------------------------------------

  public RelGraphXmlWriter(PrintWriter pw) {
    super(pw);
    xmlOutput = new XmlOutput(pw);
    xmlOutput.setGlob(true);
    xmlOutput.setCompact(false);
  }

  //~ Methods ----------------------------------------------------------------

  @Override protected void explain_(
      RelNode rel,
      List<Pair<String, @Nullable Object>> values) {
    values = replaceRexInputRef(rel, values);
    if (generic) {
      explainGeneric(rel, values);
    } else {
      explainSpecific(rel, values);
    }
  }

  /**
   * Generates generic XML (sometimes called 'element-oriented XML'). Like
   * this:
   *
   * <blockquote>
   * <code>
   * &lt;RelNode id="1" type="Join"&gt;<br>
   * &nbsp;&nbsp;&lt;Property name="condition"&gt;EMP.DEPTNO =
   * DEPT.DEPTNO&lt;/Property&gt;<br>
   * &nbsp;&nbsp;&lt;Inputs&gt;<br>
   * &nbsp;&nbsp;&nbsp;&nbsp;&lt;RelNode id="2" type="Project"&gt;<br>
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;Property name="expr1"&gt;x +
   * y&lt;/Property&gt;<br>
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;Property
   * name="expr2"&gt;45&lt;/Property&gt;<br>
   * &nbsp;&nbsp;&nbsp;&nbsp;&lt;/RelNode&gt;<br>
   * &nbsp;&nbsp;&nbsp;&nbsp;&lt;RelNode id="3" type="TableAccess"&gt;<br>
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;Property
   * name="table"&gt;SALES.EMP&lt;/Property&gt;<br>
   * &nbsp;&nbsp;&nbsp;&nbsp;&lt;/RelNode&gt;<br>
   * &nbsp;&nbsp;&lt;/Inputs&gt;<br>
   * &lt;/RelNode&gt;</code>
   * </blockquote>
   *
   * @param rel    Relational expression
   * @param values List of term-value pairs
   */
  private void explainGeneric(
      RelNode rel,
      List<Pair<String, @Nullable Object>> values) {
    String relType = rel.getRelTypeName();
    xmlOutput.beginBeginTag("RelNode");
    xmlOutput.attribute("type", relType);

    xmlOutput.endBeginTag("RelNode");

    final List<RelNode> inputs = new ArrayList<>();
    for (Pair<String, @Nullable Object> pair : values) {
      if (pair.right instanceof RelNode) {
        inputs.add((RelNode) pair.right);
        continue;
      }
      if (pair.right == null) {
        continue;
      }
      xmlOutput.beginBeginTag("Property");
      xmlOutput.attribute("name", pair.left);
      xmlOutput.endBeginTag("Property");
      xmlOutput.cdata(pair.right.toString());
      xmlOutput.endTag("Property");
    }
    xmlOutput.beginTag("Inputs", null);
    spacer.add(2);
    for (RelNode input : inputs) {
      input.explain(this);
    }
    spacer.subtract(2);
    xmlOutput.endTag("Inputs");
    xmlOutput.endTag("RelNode");
  }

  /**
   * Generates specific XML (sometimes called 'attribute-oriented XML'). Like
   * this:
   *
   * <blockquote><pre>
   * &lt;Join condition="EMP.DEPTNO = DEPT.DEPTNO"&gt;
   *   &lt;Project expr1="x + y" expr2="42"&gt;
   *   &lt;TableAccess table="SALES.EMPS"&gt;
   * &lt;/Join&gt;
   * </pre></blockquote>
   *
   * @param rel    Relational expression
   * @param values List of term-value pairs
   */
  private void explainSpecific(
      RelNode rel,
      List<Pair<String, @Nullable Object>> values) {
    String tagName = rel.getRelTypeName();
    xmlOutput.beginBeginTag(tagName);
    xmlOutput.attribute("id", rel.getId() + "");

    for (Pair<String, @Nullable Object> value : values) {
      if (value.right instanceof RelNode) {
        continue;
      }
      xmlOutput.attribute(
          value.left,
          Objects.toString(value.right));
    }
    xmlOutput.endBeginTag(tagName);
    spacer.add(2);
    for (RelNode input : rel.getInputs()) {
      input.explain(this);
    }
    spacer.subtract(2);
  }
}
