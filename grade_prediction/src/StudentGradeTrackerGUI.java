import javax.swing.*;
import javax.swing.border.*;
import javax.swing.table.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.font.*;
import java.util.*;
import java.util.List;
import java.util.stream.*;
import javax.swing.Timer;


/**
 * ╔══════════════════════════════════════════════════════════╗
 *  Student Grade Tracker  –  GUI + ML Prediction Engine
 *  Pure Java 8+, zero external dependencies
 *  Compile:  javac StudentGradeTrackerGUI.java
 *  Run:      java  StudentGradeTrackerGUI
 * ╚══════════════════════════════════════════════════════════╝
 *
 *  Prediction model: Multi-feature Linear Regression trained
 *  via Ordinary Least Squares (closed-form normal equation).
 *  Features: avg, highest, lowest, num_subjects,
 *            attendance%, study_hrs  ->  predicted final avg
 */


public class StudentGradeTrackerGUI extends JFrame {

    // ── Palette ────────────────────────────────────────────────
    static final Color BG      = new Color(0xF7F6F2);
    static final Color SURFACE = new Color(0xFFFFFF);
    static final Color SIDEBAR = new Color(0x1A1A18);
    static final Color ACCENT  = new Color(0x5B6AF0);
    static final Color TEXT1   = new Color(0x1A1A18);
    static final Color TEXT2   = new Color(0x6B6B65);
    static final Color TEXT3   = new Color(0xAAAAAA);
    static final Color BORDER  = new Color(0xE5E3DC);
    static final Color GREEN   = new Color(0x22C55E);
    static final Color AMBER   = new Color(0xF59E0B);
    static final Color RED     = new Color(0xEF4444);
    static final Color BLUE    = new Color(0x3B82F6);
    static final Color PURPLE  = new Color(0x8B5CF6);
    static final Color TEAL    = new Color(0x14B8A6);

    // ── Fonts ──────────────────────────────────────────────────
    static final Font FD  = new Font("Georgia",   Font.BOLD,  22);
    static final Font FB  = new Font("SansSerif", Font.PLAIN, 13);
    static final Font FBD = new Font("SansSerif", Font.BOLD,  13);
    static final Font FS  = new Font("SansSerif", Font.PLAIN, 11);
    static final Font FSB = new Font("SansSerif", Font.BOLD,  11);
    static final Font FBG = new Font("Georgia",   Font.BOLD,  26);

    static final String[] SUBJECTS = {
            "Mathematics","Science","English","History",
            "Computer Sci","Hindi","Geography","Art","Physical Ed","Music"
    };

    // ═══════════════════════════════════════════════════════════
    //  DATA MODEL
    // ═══════════════════════════════════════════════════════════
    static class Student {
        String fn, ln, id, sec;
        Map<String,Double> scores = new LinkedHashMap<>();
        int att  = 85;   // attendance %
        int hrs  = 10;   // study hrs/week
        Student(String fn,String ln,String id,String sec){ this.fn=fn;this.ln=ln;this.id=id;this.sec=sec; }
        double avg()  { return scores.isEmpty()?0:scores.values().stream().mapToDouble(d->d).average().orElse(0); }
        double high() { return scores.values().stream().mapToDouble(d->d).max().orElse(0); }
        double low()  { return scores.values().stream().mapToDouble(d->d).min().orElse(0); }
        String grade(){ double a=avg(); if(a>=90)return"A";if(a>=80)return"B";if(a>=65)return"C";if(a>=50)return"D";return"F"; }
        String name() { return fn+" "+ln; }
    }

    // ═══════════════════════════════════════════════════════════
    //  ML MODEL  – Ordinary Least Squares Linear Regression
    //  Θ = (XᵀX)⁻¹ Xᵀy
    //  Features: [1, avg, high, low, n_subjects, attendance, study_hrs]
    // ═══════════════════════════════════════════════════════════
    static class OLSModel {
        double[] theta;
        boolean  trained = false;
        double   rmse = 0, r2 = 0;

        // feature vector for a student
        static double[] feat(Student s){
            return new double[]{1, s.avg(), s.high(), s.low(),
                    s.scores.size(), s.att, s.hrs};
        }
        // synthetic future score label (realistic simulation)
        static double label(Student s, int seed){
            Random r=new Random((long)seed*31+7);
            double base = s.avg();
            double attBonus   = (s.att  - 75) * 0.09;
            double studyBonus = (s.hrs  - 10) * 0.14;
            double noise      = r.nextGaussian() * 2.8;
            return Math.min(100, Math.max(0, base + attBonus + studyBonus + noise));
        }

        void train(List<Student> data){
            if(data.size()<3){ trained=false; return; }
            int n=data.size(), k=7;
            double[][] X = new double[n][k];
            double[]   y = new double[n];
            for(int i=0;i<n;i++){ X[i]=feat(data.get(i)); y[i]=label(data.get(i),i); }
            double[][] XtX = mul(tp(X),X);
            double[]   Xty = mv(tp(X),y);
            double[][] inv = inv(XtX);
            if(inv==null){ trained=false; return; }
            theta=mv(inv,Xty); trained=true;
            // metrics
            double yBar=0, ss=0, sr=0;
            for(double v:y) yBar+=v; yBar/=n;
            for(int i=0;i<n;i++){
                double p=dot(theta,X[i]);
                ss+=(y[i]-yBar)*(y[i]-yBar);
                sr+=(y[i]-p)*(y[i]-p);
            }
            rmse=Math.sqrt(sr/n);
            r2 = ss==0?0:1-sr/ss;
        }

        double predict(Student s){
            if(!trained) return s.avg();
            return Math.min(100,Math.max(0,dot(theta,feat(s))));
        }
        String predGrade(double p){ if(p>=90)return"A";if(p>=80)return"B";if(p>=65)return"C";if(p>=50)return"D";return"F"; }
        double confidence(){ return Math.max(40,Math.min(96,96-rmse*1.8)); }

        // ── Matrix math ─────────────────────────────────────────
        static double dot(double[] a,double[] b){ double s=0;for(int i=0;i<a.length;i++)s+=a[i]*b[i];return s; }
        static double[][] tp(double[][] M){
            int r=M.length,c=M[0].length; double[][] T=new double[c][r];
            for(int i=0;i<r;i++) for(int j=0;j<c;j++) T[j][i]=M[i][j]; return T;
        }
        static double[][] mul(double[][] A,double[][] B){
            int m=A.length,n=B[0].length,p=B.length; double[][] C=new double[m][n];
            for(int i=0;i<m;i++) for(int j=0;j<n;j++) for(int k=0;k<p;k++) C[i][j]+=A[i][k]*B[k][j]; return C;
        }
        static double[] mv(double[][] A,double[] v){
            int m=A.length,n=v.length; double[] r=new double[m];
            for(int i=0;i<m;i++) for(int j=0;j<n;j++) r[i]+=A[i][j]*v[j]; return r;
        }
        static double[][] inv(double[][] M){  // Gauss-Jordan
            int n=M.length; double[][] a=new double[n][2*n];
            for(int i=0;i<n;i++){ for(int j=0;j<n;j++) a[i][j]=M[i][j]; a[i][i+n]=1; }
            for(int col=0;col<n;col++){
                int piv=-1; double best=1e-12;
                for(int row=col;row<n;row++) if(Math.abs(a[row][col])>best){best=Math.abs(a[row][col]);piv=row;}
                if(piv<0) return null;
                double[] tmp=a[col]; a[col]=a[piv]; a[piv]=tmp;
                double d=a[col][col];
                for(int j=0;j<2*n;j++) a[col][j]/=d;
                for(int row=0;row<n;row++){
                    if(row==col) continue;
                    double f=a[row][col];
                    for(int j=0;j<2*n;j++) a[row][j]-=f*a[col][j];
                }
            }
            double[][] inv=new double[n][n];
            for(int i=0;i<n;i++) for(int j=0;j<n;j++) inv[i][j]=a[i][j+n]; return inv;
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  APP STATE
    // ═══════════════════════════════════════════════════════════
    List<Student>   students = new ArrayList<>();
    OLSModel        model    = new OLSModel();
    JPanel          content;
    CardLayout      cards;
    JPanel[]        navBtns  = new JPanel[5];
    DefaultTableModel rosterTM;
    JTable           rosterTbl;
    JTextField       searchFld;
    // dashboard widgets
    JLabel dTotal,dAvg,dHigh,dPass;
    JPanel dTop,dDist;
    // add-student widgets
    JTextField aFn,aLn,aId,aSec,aAtt,aHrs;
    JTextField[] aSc = new JTextField[SUBJECTS.length];
    JLabel       addSt;
    // prediction widgets
    JLabel mStatus,mRMSE,mR2;
    JTextField pAtt,pHrs;
    JTextField[] pSc = new JTextField[SUBJECTS.length];
    JLabel pScore,pGrade,pConf;
    JPanel pResultPanel,pBarPanel;
    // report
    JPanel reportBody;

    // ═══════════════════════════════════════════════════════════
    //  CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════
    public StudentGradeTrackerGUI(){
        setTitle("GradeTrack — AI Grade & Prediction System");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setMinimumSize(new Dimension(1140,700));
        setSize(1240,780);
        setLocationRelativeTo(null);

        JPanel root=new JPanel(new BorderLayout());
        root.setBackground(SIDEBAR);
        root.add(buildSidebar(),BorderLayout.WEST);

        cards  = new CardLayout();
        content= new JPanel(cards);
        content.setBackground(BG);
        content.add(buildDashboard(), "dash");
        content.add(buildAddPanel(),  "add");
        content.add(buildRoster(),    "roster");
        content.add(buildPredict(),   "predict");
        content.add(buildReport(),    "report");
        root.add(content,BorderLayout.CENTER);
        setContentPane(root);

        loadSample();
        setVisible(true);
    }

    // ═══════════════════════════════════════════════════════════
    //  SIDEBAR
    // ═══════════════════════════════════════════════════════════
    JPanel buildSidebar(){
        JPanel sb=new JPanel(){
            @Override protected void paintComponent(Graphics g){
                super.paintComponent(g);
                ((Graphics2D)g).setPaint(new GradientPaint(0,0,SIDEBAR,0,getHeight(),new Color(0x0D0D0C)));
                ((Graphics2D)g).fillRect(0,0,getWidth(),getHeight());
            }
        };
        sb.setPreferredSize(new Dimension(215,0));
        sb.setLayout(new BoxLayout(sb,BoxLayout.Y_AXIS));
        sb.setBorder(new EmptyBorder(26,0,26,0));

        JPanel logo=new JPanel(new FlowLayout(FlowLayout.LEFT,18,0));
        logo.setOpaque(false); logo.setMaximumSize(new Dimension(215,50));
        JLabel ico=new JLabel("◈"); ico.setFont(new Font("SansSerif",Font.PLAIN,22)); ico.setForeground(ACCENT);
        JPanel lt=new JPanel(); lt.setOpaque(false); lt.setLayout(new BoxLayout(lt,BoxLayout.Y_AXIS));
        JLabel tt=new JLabel("GradeTrack"); tt.setFont(new Font("Georgia",Font.BOLD,14)); tt.setForeground(Color.WHITE);
        JLabel tv=new JLabel("AI Edition");  tv.setFont(FS); tv.setForeground(new Color(155,155,153));
        lt.add(tt); lt.add(tv);
        logo.add(ico); logo.add(lt);
        sb.add(logo); sb.add(Box.createVerticalStrut(28));

        String[][] nav={{"⬡","Dashboard","dash"},{"⊕","Add Student","add"},
                {"≡","Roster","roster"},{"◉","AI Prediction","predict"},{"◎","Report","report"}};
        for(int i=0;i<nav.length;i++){
            final int ix=i; final String card=nav[i][2];
            navBtns[i]=navItem(nav[i][0],nav[i][1],i==0);
            navBtns[i].addMouseListener(new MouseAdapter(){
                @Override public void mouseClicked(MouseEvent e){ go(card,ix); }
                @Override public void mouseEntered(MouseEvent e){ navBtns[ix].setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)); }
            });
            sb.add(navBtns[i]); sb.add(Box.createVerticalStrut(3));
        }
        sb.add(Box.createVerticalGlue());
        JPanel footer=new JPanel(new FlowLayout(FlowLayout.LEFT,18,4)); footer.setOpaque(false); footer.setMaximumSize(new Dimension(215,32));
        JLabel fl=new JLabel("OLS Linear Regression Model"); fl.setFont(new Font("SansSerif",Font.PLAIN,10)); fl.setForeground(new Color(85,85,83));
        footer.add(fl); sb.add(footer);
        return sb;
    }

    JPanel navItem(String icon,String label,boolean active){
        JPanel p=new JPanel(new FlowLayout(FlowLayout.LEFT,18,9));
        p.setOpaque(active); p.setBackground(new Color(0x2A2A28));
        p.setMaximumSize(new Dimension(215,40));
        p.setBorder(active ? BorderFactory.createMatteBorder(0,3,0,0,ACCENT) : new EmptyBorder(0,3,0,0));
        JLabel il=new JLabel(icon); il.setFont(new Font("SansSerif",Font.PLAIN,15)); il.setForeground(active?ACCENT:new Color(125,125,123));
        JLabel ll=new JLabel(label); ll.setFont(active?FBD:FB); ll.setForeground(active?Color.WHITE:new Color(150,150,148));
        p.add(il); p.add(ll); return p;
    }

    void go(String card,int idx){
        cards.show(content,card);
        for(int i=0;i<navBtns.length;i++){
            boolean a=(i==idx);
            navBtns[i].setOpaque(a);
            navBtns[i].setBorder(a?BorderFactory.createMatteBorder(0,3,0,0,ACCENT):new EmptyBorder(0,3,0,0));
            for(Component c:navBtns[i].getComponents()) if(c instanceof JLabel){
                JLabel l=(JLabel)c;
                if(l.getText().length()<=2) l.setForeground(a?ACCENT:new Color(125,125,123));
                else{ l.setFont(a?FBD:FB); l.setForeground(a?Color.WHITE:new Color(150,150,148)); }
            }
            navBtns[i].revalidate(); navBtns[i].repaint();
        }
        if("dash".equals(card))    refreshDash();
        if("roster".equals(card))  refreshRoster();
        if("report".equals(card))  refreshReport();
        if("predict".equals(card)) refreshPred();
    }

    // ═══════════════════════════════════════════════════════════
    //  DASHBOARD
    // ═══════════════════════════════════════════════════════════
    JPanel buildDashboard(){
        JPanel p=new JPanel(new BorderLayout()); p.setBackground(BG);
        p.add(hdr("Dashboard","Class overview and AI prediction summary"),BorderLayout.NORTH);
        JPanel body=new JPanel(); body.setLayout(new BoxLayout(body,BoxLayout.Y_AXIS));
        body.setBackground(BG); body.setBorder(new EmptyBorder(0,24,24,24));
        JPanel row=new JPanel(new GridLayout(1,4,13,0)); row.setOpaque(false); row.setMaximumSize(new Dimension(Integer.MAX_VALUE,106));
        dTotal=new JLabel("0"); dAvg=new JLabel("0.0"); dHigh=new JLabel("0.0"); dPass=new JLabel("0%");
        row.add(statCard("Total Students",dTotal,"enrolled",ACCENT));
        row.add(statCard("Class Average",  dAvg,  "out of 100",GREEN));
        row.add(statCard("Highest Average",dHigh, "top performer",BLUE));
        row.add(statCard("Pass Rate",      dPass, "average >= 50",PURPLE));
        body.add(row); body.add(Box.createVerticalStrut(16));
        JPanel bot=new JPanel(new GridLayout(1,2,14,0)); bot.setOpaque(false);
        dTop=new JPanel(); dTop.setLayout(new BoxLayout(dTop,BoxLayout.Y_AXIS)); dTop.setBackground(SURFACE); dTop.setBorder(cb());
        dDist=new JPanel(); dDist.setLayout(new BoxLayout(dDist,BoxLayout.Y_AXIS)); dDist.setBackground(SURFACE); dDist.setBorder(cb());
        bot.add(dTop); bot.add(dDist); body.add(bot);
        p.add(sw(body),BorderLayout.CENTER); return p;
    }

    void refreshDash(){
        if(students.isEmpty()){
            dTotal.setText("0"); dAvg.setText("0.0"); dHigh.setText("0.0"); dPass.setText("0%");
            emptyPanel(dTop,"Top Performers","No students yet.");
            emptyPanel(dDist,"Grade Distribution","No data."); return;
        }
        double[] avgs=students.stream().mapToDouble(Student::avg).toArray();
        double ca=Arrays.stream(avgs).average().orElse(0);
        double hi=Arrays.stream(avgs).max().orElse(0);
        long ps=Arrays.stream(avgs).filter(a->a>=50).count();
        dTotal.setText(String.valueOf(students.size()));
        dAvg.setText(String.format("%.1f",ca));
        dHigh.setText(String.format("%.1f",hi));
        dPass.setText((int)Math.round(ps*100.0/students.size())+"%");

        dTop.removeAll(); dTop.add(sec("Top Performers")); dTop.add(Box.createVerticalStrut(12));
        students.stream().sorted((a,b)->Double.compare(b.avg(),a.avg())).limit(5).forEach(s->{ dTop.add(topRow(s)); dTop.add(Box.createVerticalStrut(8)); });
        dTop.add(Box.createVerticalGlue()); dTop.revalidate(); dTop.repaint();

        dDist.removeAll(); dDist.add(sec("Grade Distribution")); dDist.add(Box.createVerticalStrut(14));
        long[] gc={ Arrays.stream(avgs).filter(a->a>=90).count(), Arrays.stream(avgs).filter(a->a>=80&&a<90).count(),
                Arrays.stream(avgs).filter(a->a>=65&&a<80).count(), Arrays.stream(avgs).filter(a->a>=50&&a<65).count(),
                Arrays.stream(avgs).filter(a->a<50).count() };
        String[] gl={"A (90-100)","B (80-89)","C (65-79)","D (50-64)","F (0-49)"};
        Color[]  gcl={GREEN,BLUE,AMBER,new Color(0xFF7849),RED};
        for(int i=0;i<5;i++){ dDist.add(gbar(gl[i],(int)gc[i],students.size(),gcl[i])); dDist.add(Box.createVerticalStrut(7)); }
        dDist.add(Box.createVerticalGlue()); dDist.revalidate(); dDist.repaint();
    }

    // ═══════════════════════════════════════════════════════════
    //  ADD STUDENT
    // ═══════════════════════════════════════════════════════════
    JPanel buildAddPanel(){
        JPanel p=new JPanel(new BorderLayout()); p.setBackground(BG);
        p.add(hdr("Add Student","Enter details, scores and study habits (used by AI model)"),BorderLayout.NORTH);
        JPanel body=new JPanel(); body.setLayout(new BoxLayout(body,BoxLayout.Y_AXIS));
        body.setBackground(BG); body.setBorder(new EmptyBorder(0,24,24,24));
        JPanel card=card();
        JPanel f1=new JPanel(new GridLayout(2,2,12,12)); f1.setOpaque(false); f1.setMaximumSize(new Dimension(Integer.MAX_VALUE,116));
        aFn=tf("First name"); aLn=tf("Last name"); aId=tf("Student ID"); aSec=tf("Section");
        f1.add(lf("First Name",aFn)); f1.add(lf("Last Name",aLn)); f1.add(lf("Student ID",aId)); f1.add(lf("Section",aSec));
        card.add(f1); card.add(Box.createVerticalStrut(14));

        JLabel predLbl=sec("AI Model Inputs"); predLbl.setForeground(TEAL); predLbl.setAlignmentX(Component.LEFT_ALIGNMENT);
        card.add(predLbl); card.add(Box.createVerticalStrut(8));
        JPanel f2=new JPanel(new GridLayout(1,2,12,0)); f2.setOpaque(false); f2.setMaximumSize(new Dimension(Integer.MAX_VALUE,64));
        aAtt=tf("85"); aHrs=tf("10");
        f2.add(lf("Attendance % (0-100)",aAtt)); f2.add(lf("Study Hours/Week (0-40)",aHrs));
        card.add(f2); card.add(Box.createVerticalStrut(16));

        card.add(sec("Subject Scores (0–100, blank = skip)")); card.add(Box.createVerticalStrut(8));
        JPanel sg=new JPanel(new GridLayout(2,5,11,11)); sg.setOpaque(false); sg.setMaximumSize(new Dimension(Integer.MAX_VALUE,108));
        for(int i=0;i<SUBJECTS.length;i++){ aSc[i]=tf("—"); aSc[i].setHorizontalAlignment(JTextField.CENTER); sg.add(lf(SUBJECTS[i],aSc[i])); }
        card.add(sg); card.add(Box.createVerticalStrut(18));

        addSt=new JLabel(" "); addSt.setFont(FB); addSt.setAlignmentX(Component.LEFT_ALIGNMENT);
        JPanel br=new JPanel(new FlowLayout(FlowLayout.LEFT,0,0)); br.setOpaque(false); br.setMaximumSize(new Dimension(Integer.MAX_VALUE,40));
        JButton addB=pBtn("Add Student"); JButton clrB=sBtn("Clear"); JButton smpB=sBtn("Load Sample Data");
        addB.addActionListener(e->doAdd()); clrB.addActionListener(e->clearAdd());
        smpB.addActionListener(e->{ loadSample(); stat(addSt,"10 sample students loaded.",true); });
        br.add(addB); br.add(Box.createHorizontalStrut(10)); br.add(clrB); br.add(Box.createHorizontalStrut(10)); br.add(smpB);
        card.add(addSt); card.add(Box.createVerticalStrut(8)); card.add(br);
        body.add(card);
        p.add(sw(body),BorderLayout.CENTER); return p;
    }

    void doAdd(){
        String fn=aFn.getText().trim(), ln=aLn.getText().trim(), id=aId.getText().trim(), sc=aSec.getText().trim();
        if(fn.isEmpty()||ln.isEmpty()){ stat(addSt,"Please enter first and last name.",false); return; }
        if(id.isEmpty()) id="S"+String.format("%03d",students.size()+1);
        final String fid=id;
        if(students.stream().anyMatch(s->s.id.equalsIgnoreCase(fid))){ stat(addSt,"Student ID already exists.",false); return; }
        int att=85,hrs=10;
        try{ att=Math.min(100,Math.max(0,Integer.parseInt(aAtt.getText().trim()))); }catch(Exception ex){}
        try{ hrs=Math.min(40, Math.max(0,Integer.parseInt(aHrs.getText().trim()))); }catch(Exception ex){}
        Student s=new Student(fn,ln,id,sc.isEmpty()?"—":sc); s.att=att; s.hrs=hrs;
        boolean any=false;
        for(int i=0;i<SUBJECTS.length;i++){
            String v=aSc[i].getText().trim();
            if(!v.isEmpty()){ try{ double d=Double.parseDouble(v); if(d<0||d>100){stat(addSt,SUBJECTS[i]+" must be 0-100.",false);return;} s.scores.put(SUBJECTS[i],d);any=true; }catch(Exception ex){stat(addSt,"Invalid score for "+SUBJECTS[i],false);return;} }
        }
        if(!any){ stat(addSt,"Enter at least one subject score.",false); return; }
        students.add(s); model.train(students);
        stat(addSt,"✓ "+s.name()+" added. Avg="+String.format("%.1f",s.avg())+" Grade="+s.grade(),true);
        clearAdd();
    }

    void clearAdd(){ aFn.setText(""); aLn.setText(""); aId.setText(""); aSec.setText(""); aAtt.setText(""); aHrs.setText(""); for(JTextField f:aSc)f.setText(""); }

    // ═══════════════════════════════════════════════════════════
    //  ROSTER
    // ═══════════════════════════════════════════════════════════
    JPanel buildRoster(){
        JPanel p=new JPanel(new BorderLayout()); p.setBackground(BG);
        p.add(hdr("Student Roster","Search, view and manage students — AI trend shown"),BorderLayout.NORTH);
        JPanel body=new JPanel(new BorderLayout(0,13)); body.setBackground(BG); body.setBorder(new EmptyBorder(0,24,24,24));
        JPanel tb=new JPanel(new BorderLayout(10,0)); tb.setOpaque(false);
        searchFld=tf("Search name or ID…");
        searchFld.getDocument().addDocumentListener(new javax.swing.event.DocumentListener(){
            public void insertUpdate(javax.swing.event.DocumentEvent e){refreshRoster();}
            public void removeUpdate(javax.swing.event.DocumentEvent e){refreshRoster();}
            public void changedUpdate(javax.swing.event.DocumentEvent e){refreshRoster();}
        });
        tb.add(searchFld,BorderLayout.CENTER);
        JButton del=dBtn("Remove Selected"); del.addActionListener(e->delSel()); tb.add(del,BorderLayout.EAST);
        body.add(tb,BorderLayout.NORTH);
        String[] cols={"Name","ID","Section","Att%","Hrs/wk","Avg","High","Low","Grade","AI Predicted","Trend"};
        rosterTM=new DefaultTableModel(cols,0){public boolean isCellEditable(int r,int c){return false;}};
        rosterTbl=new JTable(rosterTM); styleTable(rosterTbl);
        int[] ws={145,65,65,45,55,55,55,55,55,95,65};
        for(int i=0;i<ws.length;i++) rosterTbl.getColumnModel().getColumn(i).setPreferredWidth(ws[i]);
        rosterTbl.getColumnModel().getColumn(8).setCellRenderer(gradeRender());
        rosterTbl.getColumnModel().getColumn(9).setCellRenderer(gradeRender());
        rosterTbl.getColumnModel().getColumn(10).setCellRenderer((t,v,sel,foc,r,c)->{
            JLabel l=(JLabel)new DefaultTableCellRenderer().getTableCellRendererComponent(t,v,sel,foc,r,c);
            l.setHorizontalAlignment(JLabel.CENTER); l.setFont(FBD); String s=v+"";
            l.setForeground(s.startsWith("▲")?GREEN:s.startsWith("▼")?RED:AMBER);
            if(sel)l.setBackground(new Color(0xEEF0FF)); return l;});
        JScrollPane sc=new JScrollPane(rosterTbl); sc.setBorder(new LineBorder(BORDER,1,true)); sc.getViewport().setBackground(SURFACE);
        body.add(sc,BorderLayout.CENTER); p.add(body,BorderLayout.CENTER); return p;
    }

    void refreshRoster(){
        rosterTM.setRowCount(0);
        String q=searchFld==null?"":searchFld.getText().toLowerCase();
        for(Student s:students){
            if(!q.isEmpty()&&!s.name().toLowerCase().contains(q)&&!s.id.toLowerCase().contains(q)) continue;
            double pred=model.predict(s), diff=pred-s.avg();
            String trend=diff>2?"▲ +"+String.format("%.1f",diff):diff<-2?"▼ -"+String.format("%.1f",Math.abs(diff)):"● Stable";
            rosterTM.addRow(new Object[]{s.name(),s.id,s.sec,s.att,s.hrs,
                    String.format("%.1f",s.avg()),String.format("%.1f",s.high()),String.format("%.1f",s.low()),
                    s.grade(),String.format("%.1f",pred)+" ("+model.predGrade(pred)+")",trend});
        }
    }

    void delSel(){
        int row=rosterTbl.getSelectedRow(); if(row<0){ JOptionPane.showMessageDialog(this,"Select a row first.","No Selection",JOptionPane.WARNING_MESSAGE); return; }
        String name=rosterTM.getValueAt(row,0).toString(), id=rosterTM.getValueAt(row,1).toString();
        if(JOptionPane.showConfirmDialog(this,"Remove "+name+"?","Confirm",JOptionPane.YES_NO_OPTION)==JOptionPane.YES_OPTION){
            students.removeIf(s->s.id.equals(id)); model.train(students); refreshRoster();
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  AI PREDICTION PANEL
    // ═══════════════════════════════════════════════════════════
    JPanel buildPredict(){
        JPanel p=new JPanel(new BorderLayout()); p.setBackground(BG);
        p.add(hdr("AI Prediction Engine","Ordinary Least Squares — predict future performance from current data"),BorderLayout.NORTH);
        JPanel body=new JPanel(); body.setLayout(new BoxLayout(body,BoxLayout.Y_AXIS));
        body.setBackground(BG); body.setBorder(new EmptyBorder(0,24,24,24));

        // ── Model info card ───────────────────────────────────
        JPanel info=card();
        JPanel iRow=new JPanel(new BorderLayout(14,0)); iRow.setOpaque(false); iRow.setMaximumSize(new Dimension(Integer.MAX_VALUE,56));
        JLabel mico=new JLabel("◉"); mico.setFont(new Font("SansSerif",Font.PLAIN,22)); mico.setForeground(TEAL);
        JPanel mt=new JPanel(); mt.setOpaque(false); mt.setLayout(new BoxLayout(mt,BoxLayout.Y_AXIS));
        JLabel mname=new JLabel("Linear Regression (OLS) — Normal Equation"); mname.setFont(FBD); mname.setForeground(TEXT1);
        mStatus=new JLabel("Status: not trained"); mStatus.setFont(FS); mStatus.setForeground(TEXT3);
        mt.add(mname); mt.add(mStatus);
        JPanel ml=new JPanel(new FlowLayout(FlowLayout.LEFT,10,0)); ml.setOpaque(false); ml.add(mico); ml.add(mt);
        iRow.add(ml,BorderLayout.CENTER);
        JPanel mr=new JPanel(new GridLayout(1,2,12,0)); mr.setOpaque(false); mr.setPreferredSize(new Dimension(210,46));
        JPanel rCard=miniMet("RMSE","—",AMBER); JPanel r2Card=miniMet("R² Score","—",TEAL);
        mRMSE=mmv(rCard); mR2=mmv(r2Card); mr.add(rCard); mr.add(r2Card); iRow.add(mr,BorderLayout.EAST);
        info.add(iRow); info.add(Box.createVerticalStrut(12));
        JLabel feat=new JLabel("Features: avg_score · highest · lowest · n_subjects · attendance% · study_hrs/week");
        feat.setFont(FS); feat.setForeground(TEXT2); feat.setAlignmentX(Component.LEFT_ALIGNMENT);
        info.add(feat); info.add(Box.createVerticalStrut(10));
        JButton trainB=pBtn("Retrain Model");
        trainB.addActionListener(e->{
            model.train(students); updateMStatus();
            JOptionPane.showMessageDialog(this, model.trained
                    ? "Model trained on "+students.size()+" students\nRMSE: "+String.format("%.2f",model.rmse)+"   R²: "+String.format("%.3f",model.r2)
                    : "Need at least 3 students.","Training",JOptionPane.INFORMATION_MESSAGE);
        });
        info.add(trainB);
        body.add(info); body.add(Box.createVerticalStrut(14));

        // ── Single prediction card ────────────────────────────
        JPanel pred=card();
        pred.add(sec("Predict for a Student")); pred.add(Box.createVerticalStrut(14));

        JPanel sr=new JPanel(new GridLayout(1,3,12,0)); sr.setOpaque(false); sr.setMaximumSize(new Dimension(Integer.MAX_VALUE,68));
        pAtt=tf("85"); pHrs=tf("10");
        sr.add(lf("Attendance % (0-100)",pAtt)); sr.add(lf("Study Hours/Week (0-40)",pHrs));
        JPanel pickW=new JPanel(new BorderLayout(0,4)); pickW.setOpaque(false);
        JLabel pl=new JLabel("OR PICK EXISTING"); pl.setFont(new Font("SansSerif",Font.BOLD,10)); pl.setForeground(TEXT2);
        JButton pickB=sBtn("Select Existing Student"); pickB.addActionListener(e->pickStudentPred());
        pickW.add(pl,BorderLayout.NORTH); pickW.add(pickB,BorderLayout.CENTER);
        sr.add(pickW);
        pred.add(sr); pred.add(Box.createVerticalStrut(12));

        pred.add(sec("Subject Scores (0–100)")); pred.add(Box.createVerticalStrut(8));
        JPanel pg=new JPanel(new GridLayout(2,5,11,11)); pg.setOpaque(false); pg.setMaximumSize(new Dimension(Integer.MAX_VALUE,108));
        for(int i=0;i<SUBJECTS.length;i++){ pSc[i]=tf("—"); pSc[i].setHorizontalAlignment(JTextField.CENTER); pg.add(lf(SUBJECTS[i],pSc[i])); }
        pred.add(pg); pred.add(Box.createVerticalStrut(14));

        JButton runB=pBtn("Run Prediction  ▶");
        runB.addActionListener(e->runPred());
        pred.add(runB); pred.add(Box.createVerticalStrut(14));

        pResultPanel=new JPanel(); pResultPanel.setLayout(new BoxLayout(pResultPanel,BoxLayout.Y_AXIS));
        pResultPanel.setOpaque(false); pResultPanel.setVisible(false);

        JSeparator sep=new JSeparator(); sep.setMaximumSize(new Dimension(Integer.MAX_VALUE,1)); sep.setForeground(BORDER);
        pResultPanel.add(sep); pResultPanel.add(Box.createVerticalStrut(14));

        JPanel resRow=new JPanel(new GridLayout(1,3,13,0)); resRow.setOpaque(false); resRow.setMaximumSize(new Dimension(Integer.MAX_VALUE,90));
        pScore=new JLabel("—"); pGrade=new JLabel("—"); pConf=new JLabel("—");
        resRow.add(statCard("Predicted Score",pScore,"future exam avg",TEAL));
        resRow.add(statCard("Predicted Grade",pGrade,"letter grade",   PURPLE));
        resRow.add(statCard("Confidence",     pConf, "model estimate", BLUE));
        pResultPanel.add(resRow); pResultPanel.add(Box.createVerticalStrut(14));

        pBarPanel=new JPanel(); pBarPanel.setLayout(new BoxLayout(pBarPanel,BoxLayout.Y_AXIS)); pBarPanel.setOpaque(false);
        pResultPanel.add(pBarPanel);
        pred.add(pResultPanel);
        body.add(pred); body.add(Box.createVerticalStrut(14));

        // ── Bulk predictions card ─────────────────────────────
        JPanel bulk=card();
        bulk.add(sec("Bulk Predictions — All Students")); bulk.add(Box.createVerticalStrut(10));
        JButton bulkB=sBtn("Generate Bulk Predictions");
        bulkB.addActionListener(e->showBulk(bulk,bulkB));
        bulk.add(bulkB);
        body.add(bulk);

        p.add(sw(body),BorderLayout.CENTER); return p;
    }

    void runPred(){
        int att=85,hrs=10;
        try{ att=Math.min(100,Math.max(0,Integer.parseInt(pAtt.getText().trim()))); }catch(Exception ex){}
        try{ hrs=Math.min(40, Math.max(0,Integer.parseInt(pHrs.getText().trim()))); }catch(Exception ex){}
        Student dummy=new Student("Preview","Student","PREV","—"); dummy.att=att; dummy.hrs=hrs;
        boolean any=false;
        for(int i=0;i<SUBJECTS.length;i++){
            String v=pSc[i].getText().trim();
            if(!v.isEmpty()){ try{ dummy.scores.put(SUBJECTS[i],Math.min(100,Math.max(0,Double.parseDouble(v)))); any=true; }catch(Exception ex){} }
        }
        if(!any){ JOptionPane.showMessageDialog(this,"Enter at least one score.","Input Required",JOptionPane.WARNING_MESSAGE); return; }
        if(!model.trained) model.train(students);

        double pred=model.predict(dummy), curr=dummy.avg();
        String pg=model.predGrade(pred);
        double conf=model.trained ? model.confidence() : 50;

        pScore.setText(String.format("%.1f",pred)); pScore.setForeground(bColor(pred));
        pGrade.setText(pg); pGrade.setForeground(gColor(pg));
        pConf.setText(String.format("%.0f%%",conf));

        pBarPanel.removeAll();
        pBarPanel.add(sec("Score Breakdown")); pBarPanel.add(Box.createVerticalStrut(8));
        pBarPanel.add(lbar("Current Average", curr,bColor(curr))); pBarPanel.add(Box.createVerticalStrut(4));
        pBarPanel.add(lbar("AI Predicted Score",pred,TEAL));       pBarPanel.add(Box.createVerticalStrut(12));
        pBarPanel.add(sec("Subjects")); pBarPanel.add(Box.createVerticalStrut(7));
        for(Map.Entry<String,Double> e:dummy.scores.entrySet()){
            pBarPanel.add(lbar(e.getKey(),e.getValue(),bColor(e.getValue())));
            pBarPanel.add(Box.createVerticalStrut(4));
        }
        pBarPanel.add(Box.createVerticalStrut(12));

        // Recommendation box
        JPanel rec=new JPanel(); rec.setLayout(new BoxLayout(rec,BoxLayout.Y_AXIS));
        rec.setBackground(new Color(0xEEF0FF)); rec.setMaximumSize(new Dimension(Integer.MAX_VALUE,130));
        rec.setAlignmentX(Component.LEFT_ALIGNMENT);
        rec.setBorder(new CompoundBorder(new LineBorder(new Color(0xC7D2FE),1,true),new EmptyBorder(12,14,12,14)));
        JLabel rt=new JLabel("◈  AI Recommendation"); rt.setFont(FBD); rt.setForeground(ACCENT); rt.setAlignmentX(Component.LEFT_ALIGNMENT);
        String rtext=recommendation(curr,pred,att,hrs);
        JLabel rb=new JLabel("<html><body style='width:430px'>"+rtext+"</body></html>"); rb.setFont(FB); rb.setForeground(TEXT1); rb.setAlignmentX(Component.LEFT_ALIGNMENT);
        rec.add(rt); rec.add(Box.createVerticalStrut(6)); rec.add(rb);
        pBarPanel.add(rec);
        pBarPanel.revalidate(); pBarPanel.repaint();
        pResultPanel.setVisible(true);
    }

    String recommendation(double curr,double pred,int att,int hrs){
        StringBuilder sb=new StringBuilder();
        if(pred>curr+3)      sb.append("Model predicts <b>improvement ▲</b>. ");
        else if(pred<curr-3) sb.append("Model predicts <b>decline ▼</b>. ");
        else                 sb.append("Performance predicted to remain <b>stable</b>. ");
        if(att<70)       sb.append("⚠ Attendance critically low — improving it has the strongest model weight. ");
        else if(att<80)  sb.append("Attendance could be better. ");
        else             sb.append("Good attendance. ");
        if(hrs<6)        sb.append("Increasing weekly study hours will significantly boost the prediction. ");
        else if(hrs>22)  sb.append("Study hours high — focus on quality over quantity. ");
        else             sb.append("Study schedule looks balanced. ");
        if(curr<50)      sb.append("<b>Immediate academic intervention recommended.</b>");
        else if(curr<65) sb.append("Target weaker subjects to move into the B range.");
        else if(curr>=90)sb.append("Outstanding — maintain consistency!");
        return sb.toString();
    }

    void pickStudentPred(){
        if(students.isEmpty()){ JOptionPane.showMessageDialog(this,"No students loaded.","Empty",JOptionPane.WARNING_MESSAGE); return; }
        String[] names=students.stream().map(s->s.name()+" ["+s.id+"]").toArray(String[]::new);
        String sel=(String)JOptionPane.showInputDialog(this,"Select:","Pick Student",JOptionPane.PLAIN_MESSAGE,null,names,names[0]);
        if(sel==null) return;
        String id=sel.replaceAll(".*\\[(.*)\\]","$1");
        students.stream().filter(s->s.id.equals(id)).findFirst().ifPresent(s->{
            pAtt.setText(String.valueOf(s.att)); pHrs.setText(String.valueOf(s.hrs));
            for(int i=0;i<SUBJECTS.length;i++){ Double v=s.scores.get(SUBJECTS[i]); pSc[i].setText(v!=null?String.format("%.0f",v):""); }
        });
    }

    void showBulk(JPanel bulk,JButton btn){
        // remove previous results
        Component[] comps=bulk.getComponents();
        for(int i=comps.length-1;i>1;i--) bulk.remove(comps[i]);
        if(students.isEmpty()){ JOptionPane.showMessageDialog(this,"No students.","Empty",JOptionPane.WARNING_MESSAGE); return; }
        if(!model.trained) model.train(students);
        bulk.add(Box.createVerticalStrut(12));
        String[] cols={"Name","Current Avg","AI Predicted","Δ Change","Grade","AI Grade","Trend","Recommendation"};
        DefaultTableModel dm=new DefaultTableModel(cols,0){public boolean isCellEditable(int r,int c){return false;}};
        for(Student s:students){
            double cur=s.avg(), pr=model.predict(s), d=pr-cur;
            String rec=d>5?"Likely to improve":d<-5?"At risk of decline":"Stable trajectory";
            dm.addRow(new Object[]{s.name(),String.format("%.1f",cur),String.format("%.1f",pr),String.format("%+.1f",d),s.grade(),model.predGrade(pr),d>=0?"▲":"▼",rec});
        }
        JTable bt=new JTable(dm); styleTable(bt);
        bt.getColumnModel().getColumn(3).setCellRenderer((t,v,sel,foc,r,c)->{
            JLabel l=(JLabel)new DefaultTableCellRenderer().getTableCellRendererComponent(t,v,sel,foc,r,c);
            l.setHorizontalAlignment(JLabel.CENTER); l.setFont(FBD);
            try{ l.setForeground(Double.parseDouble(v+"")>=0?GREEN:RED); }catch(Exception ex){}
            if(sel)l.setBackground(new Color(0xEEF0FF)); return l;});
        bt.getColumnModel().getColumn(4).setCellRenderer(gradeRender());
        bt.getColumnModel().getColumn(5).setCellRenderer(gradeRender());
        bt.getColumnModel().getColumn(6).setCellRenderer((t,v,sel,foc,r,c)->{
            JLabel l=(JLabel)new DefaultTableCellRenderer().getTableCellRendererComponent(t,v,sel,foc,r,c);
            l.setHorizontalAlignment(JLabel.CENTER); l.setFont(FBD); l.setForeground("▲".equals(v+"") ? GREEN : RED);
            if(sel)l.setBackground(new Color(0xEEF0FF)); return l;});
        bt.getColumnModel().getColumn(0).setPreferredWidth(140); bt.getColumnModel().getColumn(7).setPreferredWidth(160);
        JScrollPane sc=new JScrollPane(bt); sc.setBorder(new LineBorder(BORDER,1,true));
        sc.setMaximumSize(new Dimension(Integer.MAX_VALUE,240)); sc.setAlignmentX(Component.LEFT_ALIGNMENT);
        bulk.add(sc); bulk.revalidate(); bulk.repaint();
    }

    void refreshPred(){ updateMStatus(); }
    void updateMStatus(){
        if(mStatus==null) return;
        if(!model.trained){ mStatus.setText("Status: not trained — add ≥ 3 students"); mStatus.setForeground(AMBER); if(mRMSE!=null){mRMSE.setText("—");mR2.setText("—");} return; }
        mStatus.setText("Status: trained on "+students.size()+" students"); mStatus.setForeground(GREEN);
        if(mRMSE!=null){ mRMSE.setText(String.format("%.2f",model.rmse)); mR2.setText(String.format("%.3f",model.r2)); }
    }

    // ═══════════════════════════════════════════════════════════
    //  REPORT
    // ═══════════════════════════════════════════════════════════
    JPanel buildReport(){
        JPanel p=new JPanel(new BorderLayout()); p.setBackground(BG);
        p.add(hdr("Class Report","Full analytics with AI predictions and grade breakdown"),BorderLayout.NORTH);
        reportBody=new JPanel(); reportBody.setLayout(new BoxLayout(reportBody,BoxLayout.Y_AXIS));
        reportBody.setBackground(BG); reportBody.setBorder(new EmptyBorder(0,24,24,24));
        p.add(sw(reportBody),BorderLayout.CENTER); return p;
    }

    void refreshReport(){
        reportBody.removeAll();
        if(students.isEmpty()){ reportBody.add(muted("No students. Add students first.")); reportBody.revalidate(); return; }
        if(!model.trained) model.train(students);

        double[] avgs=students.stream().mapToDouble(Student::avg).toArray();
        double ca=Arrays.stream(avgs).average().orElse(0), hi=Arrays.stream(avgs).max().orElse(0), lo=Arrays.stream(avgs).min().orElse(0);
        Student topS=students.get(0),botS=students.get(0); double ta=0,ba=100;
        for(int i=0;i<students.size();i++){if(avgs[i]>ta){ta=avgs[i];topS=students.get(i);}if(avgs[i]<ba){ba=avgs[i];botS=students.get(i);}}

        JPanel cs=new JPanel(new GridLayout(1,4,13,0)); cs.setOpaque(false); cs.setMaximumSize(new Dimension(Integer.MAX_VALUE,106));
        JLabel tv=new JLabel(String.valueOf(students.size())), av=new JLabel(String.format("%.1f",ca)), hv=new JLabel(String.format("%.1f",hi)), lv=new JLabel(String.format("%.1f",lo));
        cs.add(statCard("Total Students",tv,"enrolled",ACCENT)); cs.add(statCard("Class Average",av,"Grade "+grade(ca),GREEN));
        cs.add(statCard("Highest",hv,topS.name(),BLUE)); cs.add(statCard("Lowest",lv,botS.name(),RED));
        reportBody.add(cs); reportBody.add(Box.createVerticalStrut(15));

        JPanel pc=card(); pc.add(sec("Performance Ranking")); pc.add(Box.createVerticalStrut(10));
        students.stream().sorted((a,b)->Double.compare(b.avg(),a.avg())).forEach(s->{ pc.add(lbar(s.name(),s.avg(),bColor(s.avg()))); pc.add(Box.createVerticalStrut(4)); });
        reportBody.add(pc); reportBody.add(Box.createVerticalStrut(13));

        Map<String,Double> st=new LinkedHashMap<>(); Map<String,Integer> sc2=new LinkedHashMap<>();
        students.forEach(s->s.scores.forEach((k,v)->{st.merge(k,v,Double::sum);sc2.merge(k,1,Integer::sum);}));
        if(!st.isEmpty()){
            JPanel sCard=card(); sCard.add(sec("Subject Averages")); sCard.add(Box.createVerticalStrut(10));
            st.entrySet().stream().sorted((a,b)->Double.compare(b.getValue()/sc2.get(b.getKey()),a.getValue()/sc2.get(a.getKey()))).forEach(e->{
                double sa=e.getValue()/sc2.get(e.getKey()); sCard.add(lbar(e.getKey(),sa,bColor(sa))); sCard.add(Box.createVerticalStrut(4)); });
            reportBody.add(sCard); reportBody.add(Box.createVerticalStrut(13));
        }

        JPanel tc=card(); tc.add(sec("Complete Report with AI Predictions")); tc.add(Box.createVerticalStrut(10));
        String[] cols={"Name","ID","Sec","Att%","Hrs","Avg","AI Pred","Δ","Grade","AI Grade","Status"};
        DefaultTableModel dm=new DefaultTableModel(cols,0){public boolean isCellEditable(int r,int c){return false;}};
        for(Student s:students){
            double cur=s.avg(), pr=model.predict(s), d=pr-cur;
            dm.addRow(new Object[]{s.name(),s.id,s.sec,s.att,s.hrs,String.format("%.1f",cur),String.format("%.1f",pr),
                    String.format("%+.1f",d),s.grade(),model.predGrade(pr),cur>=50?"PASS":"FAIL"});
        }
        JTable tbl=new JTable(dm); styleTable(tbl);
        tbl.getColumnModel().getColumn(8).setCellRenderer(gradeRender());
        tbl.getColumnModel().getColumn(9).setCellRenderer(gradeRender());
        tbl.getColumnModel().getColumn(7).setCellRenderer((t,v,sel,foc,r,c)->{
            JLabel l=(JLabel)new DefaultTableCellRenderer().getTableCellRendererComponent(t,v,sel,foc,r,c);
            l.setHorizontalAlignment(JLabel.CENTER); l.setFont(FBD);
            try{ l.setForeground(Double.parseDouble(v+"")>=0?GREEN:RED); }catch(Exception ex){}
            if(sel)l.setBackground(new Color(0xEEF0FF)); return l;});
        tbl.getColumnModel().getColumn(10).setCellRenderer((t,v,sel,foc,r,c)->{
            JLabel l=(JLabel)new DefaultTableCellRenderer().getTableCellRendererComponent(t,v,sel,foc,r,c);
            l.setHorizontalAlignment(JLabel.CENTER); l.setFont(FBD); l.setForeground("PASS".equals(v)?GREEN:RED);
            if(sel)l.setBackground(new Color(0xEEF0FF)); return l;});
        tbl.getColumnModel().getColumn(0).setPreferredWidth(140);
        JScrollPane tsc=new JScrollPane(tbl); tsc.setBorder(new LineBorder(BORDER,1,true));
        tsc.setMaximumSize(new Dimension(Integer.MAX_VALUE,300)); tsc.setAlignmentX(Component.LEFT_ALIGNMENT);
        tc.add(tsc); reportBody.add(tc);
        reportBody.revalidate(); reportBody.repaint();
    }

    // ═══════════════════════════════════════════════════════════
    //  SAMPLE DATA
    // ═══════════════════════════════════════════════════════════
    void loadSample(){
        students.clear();
        Object[][] rows={
                {"Aarav","Gupta","S001","10-A",new double[]{88,92,75,80,95},90,14},
                {"Priya","Sharma","S002","10-A",new double[]{72,68,85,90,88},78,12},
                {"Rohit","Kumar","S003","10-B",new double[]{55,60,62,58,70},65,8},
                {"Sneha","Verma","S004","10-B",new double[]{95,98,90,93,97},98,22},
                {"Karan","Singh","S005","10-A",new double[]{40,38,45,50,65},50,5},
                {"Anjali","Patel","S006","10-C",new double[]{78,82,88,75,90},82,15},
                {"Vikram","Rao","S007","10-C",new double[]{65,70,60,72,80},72,11},
                {"Meera","Joshi","S008","10-B",new double[]{90,85,92,88,95},95,20},
                {"Arjun","Mehta","S009","10-A",new double[]{33,28,42,35,40},40,4},
                {"Divya","Nair","S010","10-C",new double[]{82,88,79,85,77},88,17},
        };
        String[] subs={"Mathematics","Science","English","History","Computer Sci"};
        for(Object[] row:rows){
            Student s=new Student((String)row[0],(String)row[1],(String)row[2],(String)row[3]);
            double[] sc=(double[])row[4]; for(int i=0;i<sc.length;i++) s.scores.put(subs[i],sc[i]);
            s.att=(int)row[5]; s.hrs=(int)row[6]; students.add(s);
        }
        model.train(students); refreshRoster(); refreshDash();
    }

    // ═══════════════════════════════════════════════════════════
    //  UI HELPERS
    // ═══════════════════════════════════════════════════════════
    JPanel hdr(String t,String sub){
        JPanel h=new JPanel(new BorderLayout()); h.setBackground(SURFACE);
        h.setBorder(new CompoundBorder(new MatteBorder(0,0,1,0,BORDER),new EmptyBorder(20,24,16,24)));
        JLabel tl=new JLabel(t); tl.setFont(FD); tl.setForeground(TEXT1);
        JLabel sl=new JLabel(sub); sl.setFont(FB); sl.setForeground(TEXT2);
        h.add(tl,BorderLayout.NORTH); h.add(sl,BorderLayout.SOUTH); return h;
    }

    JPanel card(){
        JPanel c=new JPanel(); c.setLayout(new BoxLayout(c,BoxLayout.Y_AXIS));
        c.setBackground(SURFACE); c.setAlignmentX(Component.LEFT_ALIGNMENT);
        c.setBorder(new CompoundBorder(new LineBorder(BORDER,1,true),new EmptyBorder(18,20,18,20))); return c;
    }

    CompoundBorder cb(){ return new CompoundBorder(new LineBorder(BORDER,1,true),new EmptyBorder(18,20,18,20)); }

    JPanel statCard(String label,JLabel val,String sub,Color accent){
        JPanel c=new JPanel(){
            @Override protected void paintComponent(Graphics g){
                super.paintComponent(g);
                Graphics2D g2=(Graphics2D)g;
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(accent); g2.fillRoundRect(0,0,4,getHeight(),4,4);
            }
        };
        c.setBackground(SURFACE); c.setLayout(new BoxLayout(c,BoxLayout.Y_AXIS));
        c.setBorder(new CompoundBorder(new LineBorder(BORDER,1,true),new EmptyBorder(14,18,14,18)));
        JLabel ll=new JLabel(label); ll.setFont(FS); ll.setForeground(TEXT2); ll.setAlignmentX(Component.LEFT_ALIGNMENT);
        val.setFont(FBG); val.setForeground(TEXT1); val.setAlignmentX(Component.LEFT_ALIGNMENT);
        JLabel sl=new JLabel(sub); sl.setFont(FS); sl.setForeground(TEXT3); sl.setAlignmentX(Component.LEFT_ALIGNMENT);
        c.add(ll); c.add(Box.createVerticalStrut(4)); c.add(val); c.add(sl); return c;
    }

    JPanel miniMet(String label,String val,Color accent){
        JPanel c=new JPanel(); c.setLayout(new BoxLayout(c,BoxLayout.Y_AXIS));
        c.setBackground(new Color(0xF4F3EF)); c.setBorder(new CompoundBorder(new LineBorder(BORDER,1,true),new EmptyBorder(10,12,10,12)));
        JLabel vl=new JLabel(val); vl.setFont(new Font("Georgia",Font.BOLD,18)); vl.setForeground(accent); vl.setAlignmentX(Component.LEFT_ALIGNMENT);
        JLabel ll=new JLabel(label); ll.setFont(FS); ll.setForeground(TEXT2); ll.setAlignmentX(Component.LEFT_ALIGNMENT);
        c.add(vl); c.add(ll); return c;
    }

    JLabel mmv(JPanel p){ for(Component c:p.getComponents()) if(c instanceof JLabel && ((JLabel)c).getFont().getSize()>=16) return (JLabel)c; return new JLabel(); }

    JLabel sec(String t){ JLabel l=new JLabel(t); l.setFont(FBD); l.setForeground(TEXT1); l.setAlignmentX(Component.LEFT_ALIGNMENT); return l; }
    JLabel muted(String t){ JLabel l=new JLabel(t); l.setFont(FB); l.setForeground(TEXT3); l.setAlignmentX(Component.LEFT_ALIGNMENT); return l; }

    JTextField tf(String ph){
        JTextField f=new JTextField(){
            @Override protected void paintComponent(Graphics g){
                super.paintComponent(g);
                if(getText().isEmpty()&&!isFocusOwner()){
                    Graphics2D g2=(Graphics2D)g; g2.setColor(new Color(170,168,163)); g2.setFont(getFont());
                    g2.drawString(ph,8,getHeight()/2+5);
                }
            }
        };
        f.setFont(FB); f.setForeground(TEXT1); f.setBackground(new Color(0xFAF9F5));
        f.setBorder(new CompoundBorder(new LineBorder(BORDER,1,true),new EmptyBorder(4,8,4,8)));
        f.setPreferredSize(new Dimension(0,36)); return f;
    }

    JPanel lf(String label,JComponent field){
        JPanel p=new JPanel(new BorderLayout(0,4)); p.setOpaque(false);
        JLabel l=new JLabel(label.toUpperCase()); l.setFont(new Font("SansSerif",Font.BOLD,10)); l.setForeground(TEXT2);
        p.add(l,BorderLayout.NORTH); p.add(field,BorderLayout.CENTER); return p;
    }

    JButton pBtn(String t){
        JButton b=new JButton(t){
            @Override protected void paintComponent(Graphics g){
                Graphics2D g2=(Graphics2D)g;
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(getModel().isPressed()?ACCENT.darker():getModel().isRollover()?ACCENT.brighter():ACCENT);
                g2.fillRoundRect(0,0,getWidth(),getHeight(),8,8);
                g2.setColor(Color.WHITE); g2.setFont(getFont());
                FontMetrics fm=g2.getFontMetrics();
                g2.drawString(getText(),(getWidth()-fm.stringWidth(getText()))/2,(getHeight()+fm.getAscent()-fm.getDescent())/2);
            }
        };
        b.setFont(FBD); b.setPreferredSize(new Dimension(170,38));
        b.setBorderPainted(false); b.setFocusPainted(false); b.setContentAreaFilled(false);
        b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)); return b;
    }

    JButton sBtn(String t){
        JButton b=new JButton(t); b.setFont(FB); b.setForeground(TEXT1); b.setBackground(SURFACE);
        b.setBorder(new CompoundBorder(new LineBorder(BORDER,1,true),new EmptyBorder(6,14,6,14)));
        b.setFocusPainted(false); b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        b.setPreferredSize(new Dimension(0,36)); return b;
    }

    JButton dBtn(String t){
        JButton b=new JButton(t); b.setFont(FB); b.setForeground(RED); b.setBackground(new Color(0xFFF0F0));
        b.setBorder(new CompoundBorder(new LineBorder(new Color(0xFFCCCC),1,true),new EmptyBorder(6,14,6,14)));
        b.setFocusPainted(false); b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)); return b;
    }

    void styleTable(JTable t){
        t.setFont(FB); t.setRowHeight(30); t.setShowGrid(false);
        t.setIntercellSpacing(new Dimension(0,0)); t.setBackground(SURFACE);
        t.setSelectionBackground(new Color(0xEEF0FF)); t.setSelectionForeground(TEXT1);
        t.setShowHorizontalLines(true); t.setGridColor(BORDER);
        JTableHeader h=t.getTableHeader(); h.setFont(FSB); h.setBackground(new Color(0xF2F0EA));
        h.setForeground(TEXT2); h.setBorder(new LineBorder(BORDER,1)); h.setPreferredSize(new Dimension(0,30));
        ((DefaultTableCellRenderer)h.getDefaultRenderer()).setHorizontalAlignment(JLabel.LEFT);
    }

    JScrollPane sw(JPanel p){
        JScrollPane sc=new JScrollPane(p); sc.setBorder(null);
        sc.getViewport().setBackground(BG); sc.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER); return sc;
    }

    void stat(JLabel l,String msg,boolean ok){
        l.setText(msg); l.setForeground(ok?GREEN:RED);
        new Timer(4000,e->l.setText(" ")){{ setRepeats(false); start(); }};
    }

    void emptyPanel(JPanel p,String title,String msg){
        p.removeAll(); p.add(sec(title)); p.add(Box.createVerticalStrut(10)); p.add(muted(msg));
        p.add(Box.createVerticalGlue()); p.revalidate(); p.repaint();
    }

    JPanel topRow(Student s){
        JPanel row=new JPanel(new BorderLayout(10,0)); row.setOpaque(false); row.setMaximumSize(new Dimension(Integer.MAX_VALUE,34));
        JLabel av=new JLabel(s.fn.substring(0,1)){
            @Override protected void paintComponent(Graphics g){
                Graphics2D g2=(Graphics2D)g; g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(ACCENT); g2.fillOval(0,0,getWidth(),getHeight());
                g2.setColor(Color.WHITE); g2.setFont(new Font("Georgia",Font.BOLD,12));
                FontMetrics fm=g2.getFontMetrics(); String t=getText();
                g2.drawString(t,(getWidth()-fm.stringWidth(t))/2,(getHeight()+fm.getAscent()-fm.getDescent())/2);
            }
        };
        av.setPreferredSize(new Dimension(28,28)); av.setMinimumSize(new Dimension(28,28));
        JPanel info=new JPanel(new BorderLayout()); info.setOpaque(false);
        JLabel nm=new JLabel(s.name()); nm.setFont(FBD); nm.setForeground(TEXT1);
        JLabel sid=new JLabel(s.id+" · "+s.sec); sid.setFont(FS); sid.setForeground(TEXT2);
        info.add(nm,BorderLayout.NORTH); info.add(sid,BorderLayout.SOUTH);
        JPanel right=new JPanel(new BorderLayout()); right.setOpaque(false);
        JLabel al=new JLabel(String.format("%.1f",s.avg())); al.setFont(new Font("Georgia",Font.BOLD,14)); al.setForeground(ACCENT);
        JLabel gl=new JLabel("Grade "+s.grade()); gl.setFont(FS); gl.setForeground(gColor(s.grade()));
        right.add(al,BorderLayout.NORTH); right.add(gl,BorderLayout.SOUTH);
        row.add(av,BorderLayout.WEST); row.add(info,BorderLayout.CENTER); row.add(right,BorderLayout.EAST); return row;
    }

    JPanel gbar(String label,int count,int total,Color color){
        JPanel row=new JPanel(new BorderLayout(10,0)); row.setOpaque(false); row.setMaximumSize(new Dimension(Integer.MAX_VALUE,24));
        JLabel lbl=new JLabel(label); lbl.setFont(FS); lbl.setForeground(TEXT2); lbl.setPreferredSize(new Dimension(86,18));
        double pct=total>0?(double)count/total:0;
        JPanel track=new JPanel(){
            @Override protected void paintComponent(Graphics g){
                super.paintComponent(g); Graphics2D g2=(Graphics2D)g;
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(new Color(226,224,218)); g2.fillRoundRect(0,4,getWidth(),getHeight()-8,8,8);
                int w=(int)(getWidth()*pct); if(w>0){g2.setColor(color);g2.fillRoundRect(0,4,Math.max(w,8),getHeight()-8,8,8);}
            }
        };
        track.setOpaque(false);
        JLabel cnt=new JLabel(count+" ("+Math.round(pct*100)+"%)"); cnt.setFont(FS); cnt.setForeground(TEXT2); cnt.setPreferredSize(new Dimension(54,18));
        row.add(lbl,BorderLayout.WEST); row.add(track,BorderLayout.CENTER); row.add(cnt,BorderLayout.EAST); return row;
    }

    JPanel lbar(String label,double val,Color color){
        JPanel row=new JPanel(new BorderLayout(12,0)); row.setOpaque(false); row.setMaximumSize(new Dimension(Integer.MAX_VALUE,22));
        JLabel lbl=new JLabel(label); lbl.setFont(FS); lbl.setForeground(TEXT2); lbl.setPreferredSize(new Dimension(175,18));
        double pct=val/100.0;
        JPanel track=new JPanel(){
            @Override protected void paintComponent(Graphics g){
                Graphics2D g2=(Graphics2D)g; g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(new Color(226,224,218)); g2.fillRoundRect(0,3,getWidth(),getHeight()-6,8,8);
                int w=(int)(getWidth()*pct);
                if(w>0){g2.setColor(color);g2.fillRoundRect(0,3,Math.max(w,8),getHeight()-6,8,8);
                    if(w>28){g2.setColor(Color.WHITE);g2.setFont(new Font("SansSerif",Font.BOLD,10));
                        String t=String.format("%.1f",val);g2.drawString(t,w-g2.getFontMetrics().stringWidth(t)-4,getHeight()/2+4);}}
            }
        };
        track.setOpaque(false);
        JLabel vl=new JLabel(String.format("%.1f",val)); vl.setFont(FBD); vl.setForeground(TEXT1); vl.setPreferredSize(new Dimension(34,18));
        row.add(lbl,BorderLayout.WEST); row.add(track,BorderLayout.CENTER); row.add(vl,BorderLayout.EAST); return row;
    }

    TableCellRenderer gradeRender(){
        return (t,v,sel,foc,r,c)->{
            JLabel l=(JLabel)new DefaultTableCellRenderer().getTableCellRendererComponent(t,v,sel,foc,r,c);
            l.setHorizontalAlignment(JLabel.CENTER); l.setFont(FBD); l.setForeground(gColor(v+""));
            if(sel)l.setBackground(new Color(0xEEF0FF)); return l;};
    }

    Color gColor(String g){ switch(g){case"A":return new Color(0x16A34A);case"B":return BLUE;case"C":return AMBER;case"D":return new Color(0xF97316);default:return RED;} }
    Color bColor(double a){ if(a>=90)return GREEN;if(a>=80)return BLUE;if(a>=65)return AMBER;if(a>=50)return new Color(0xF97316);return RED; }
    String grade(double a){ if(a>=90)return"A";if(a>=80)return"B";if(a>=65)return"C";if(a>=50)return"D";return"F"; }

    // ═══════════════════════════════════════════════════════════
    //  MAIN
    // ═══════════════════════════════════════════════════════════
    public static void main(String[] args){
        try{ UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); }catch(Exception e){}
        SwingUtilities.invokeLater(StudentGradeTrackerGUI::new);
    }
}
