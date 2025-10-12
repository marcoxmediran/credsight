"use client"

import type React from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AnimatedCard } from "@/components/AnimatedCard"
import { Shield, Upload, BarChart3, Brain } from "lucide-react"
import { Navigation } from "@/components/navigation"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import LiquidEther from "@/components/LiquidEther"
import ProfileCard from "@/components/ProfileCard"
import SpotlightCard from "@/components/SpotlightCard"
import MetallicText from "@/components/MetallicText"


export default function Home() {
  return (
    <div className="dark min-h-screen bg-background text-foreground relative">
      {/* LiquidEther Background - Full Page */}
      <div className="fixed inset-0 w-full h-full z-0">
        <LiquidEther
          colors={['#5227FF', '#FF9FFC', '#B19EEF']}
          mouseForce={20}
          cursorSize={100}
          isViscous={false}
          viscous={30}
          iterationsViscous={32}
          iterationsPoisson={32}
          resolution={0.5}
          isBounce={true}
          autoDemo={true}
          autoSpeed={0.5}
          autoIntensity={2.2}
          takeoverDuration={0.25}
          autoResumeDelay={0}
          autoRampDuration={0.6}
        />
      </div>

      {/* All Content */}
      <div className="relative z-10">
        <Navigation />

        {/* Hero Section */}
        <section className="flex h-screen items-center justify-center px-4">
          <AnimatedCard>
            <div className="text-center">
              <MetallicText className="text-balance font-sans text-6xl font-bold tracking-tight md:text-8xl lg:text-9xl">
                <h1>CREDSIGHT</h1>
              </MetallicText>
              <p className="mx-auto mt-6 max-w-2xl text-pretty text-lg text-muted-foreground md:text-xl">
                Advanced fraud detection powered by machine learning
              </p>
              <Link href="/analyze">
                <button className="premium-button mt-8 px-8 py-4 text-lg font-semibold">
                  Get Started
                </button>
              </Link>
              <style jsx>{`
                .premium-button {
                  background: linear-gradient(to right, #C471ED, #12C2E9);
                  color: #fff;
                  letter-spacing: 0.5px;
                  border: none;
                  border-radius: 8px;
                  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                  position: relative;
                  overflow: hidden;
                  cursor: pointer;
                  transition: all 1.5s ease-in-out;
                }
                
                .premium-button::before {
                  content: '';
                  position: absolute;
                  top: 0;
                  left: -100%;
                  width: 100%;
                  height: 100%;
                  background: linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.3),
                    transparent
                  );
                  transition: left 0.8s ease;
                }
                
                .premium-button:hover {
                  background: linear-gradient(to right, #12C2E9, #C471ED);
                  transform: scale(1.03);
                }
                
                .premium-button:hover::before {
                  left: 100%;
                }
              `}</style>
            </div>
          </AnimatedCard>
        </section>

      {/* Key Features Section */}
      <section id="features" className="px-4 py-24">
        <div className="mx-auto max-w-6xl">
          <AnimatedCard>
            <h2 className="mb-4 text-center text-3xl font-bold md:text-4xl">Key Features</h2>
            <p className="mb-16 text-center text-muted-foreground">
              Advanced comparative analysis tools for graph-based fraud detection research
            </p>
          </AnimatedCard>

          <div className="mx-auto max-w-2xl space-y-6">
            <AnimatedCard delay={0}>
              <FeatureCard
                icon={<Upload className="h-8 w-8" />}
                title="Dual Protocol CSV Processing"
                description="Upload transaction data with or without ground truth labels. Supports large datasets up to 800MB with automatic sampling and chunked processing."
                delay={0}
              />
            </AnimatedCard>
            <AnimatedCard delay={150}>
              <FeatureCard
                icon={<Brain className="h-8 w-8" />}
                title="R-GCN vs ERGCN Comparison"
                description="Side-by-side comparison of traditional R-GCN and Enhanced R-GCN models with statistical significance testing and performance metrics."
                delay={0}
              />
            </AnimatedCard>
            <AnimatedCard delay={300}>
              <FeatureCard
                icon={<BarChart3 className="h-8 w-8" />}
                title="Interactive Results Analysis"
                description="Explore predictions with advanced filtering, search functionality, and confidence details with threshold-based fraud classification."
                delay={0}
              />
            </AnimatedCard>
            <AnimatedCard delay={450}>
              <FeatureCard
                icon={<Shield className="h-8 w-8" />}
                title="Comprehensive Model Metrics"
                description="Detailed performance evaluation including Recall, F1-Score, AUC metrics, and statistical comparison with p-value analysis."
                delay={0}
              />
            </AnimatedCard>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="px-4 py-24">
        <div className="mx-auto max-w-6xl">
          <AnimatedCard>
            <h2 className="mb-4 text-center text-3xl font-bold md:text-4xl">How It Works</h2>
            <p className="mb-16 text-center text-muted-foreground">
              Three simple steps to detect fraud in your transactions
            </p>
          </AnimatedCard>

          <div className="mb-12 grid gap-6 md:grid-cols-3">
            <StepCard
              number="01"
              title="Upload Data"
              description="Upload your CSV file containing transaction data with TransactionID column"
              delay={0}
            />
            <StepCard
              number="02"
              title="ML Analysis"
              description="ERGCN model processes each transaction and generates fraud probability scores"
              delay={200}
            />
            <StepCard
              number="03"
              title="View Result"
              description="Get detailed results with filtering options and comprehensive statistics overview"
              delay={400}
            />
          </div>

          {/* GIF Container */}
          <AnimatedCard delay={600}>
            <div className="overflow-hidden rounded-lg border border-border bg-card">
              <div className="flex aspect-video items-center justify-center">
                <img
                  src="/fraud-detection-dashboard-interface.gif"
                  alt="CREDSIGHT Demo"
                  className="h-full w-full object-cover"
                />
              </div>
            </div>
          </AnimatedCard>
        </div>
      </section>

      {/* About This Project Section */}
      <section id="about" className="px-4 py-24">
        <div className="mx-auto max-w-4xl">
          <AnimatedCard>
            <h2 className="mb-8 text-center text-3xl font-bold md:text-4xl">About This Project</h2>
            <Card className="border-border bg-card">
              <CardContent className="pt-6">
                <div className="space-y-4 text-pretty leading-relaxed text-muted-foreground">
                  <p className="text-center">
                  <strong>CredSight</strong>, short for Credit Insight, embodies the project's goal of providing clearer visibility into fraudulent activities within financial transactions. The name reflects the system's capability to "see through" complex transactional relationships and uncover hidden fraud patterns using graph-based learning techniques.
                  </p>
                  <p className="text-center">
                  This undergraduate thesis project presents a comparative analysis between two graph-based machine learning models for credit card fraud detection: the Relational Graph Convolutional Network (R-GCN) and a modified version termed the Enhanced Relational Graph Convolutional Network (ERGCN). Using the IEEE-CIS Vesta Financial Fraud Detection dataset, the system evaluates how these models perform in identifying fraudulent transactions by analyzing inter-entity relationships and behavioral patterns within transaction networks.
                  </p>
                  <p className="text-center">
                  <strong>The novelty of this research lies in its pioneering application of ERGCN to the IEEE-CIS dataset</strong>, examining how enhanced relational modeling and temporal integration can influence fraud detection performance compared to conventional graph neural network approaches. This investigation provides valuable insights into the role of graph-based reasoning in large-scale financial transaction analysis.
                  </p>
                  <p className="text-center">
                  The CredSight web application offers an interactive platform for uploading transaction data and comparing model performance through key metrics such as Recall, F1-Score, and AUC. Users can explore individual transaction predictions, review confidence levels through threshold-based classification, and analyze statistical differences between the two models. The system accommodates both datasets with ground truth labels (for validation) and unlabeled datasets (for prediction).
                  </p>
                  <p className="text-center">
                  Developed by five Computer Science students as an undergraduate thesis project, CredSight demonstrates the practical application of graph neural networks in financial fraud detection while offering a robust comparative framework for evaluating model performance in real-world transactional contexts.
                  </p>
                </div>
              </CardContent>
            </Card>
          </AnimatedCard>
        </div>
      </section>

      {/* Team Profiles Section */}
      <section id="team" className="px-4 py-24">
        <div className="mx-auto max-w-6xl">
          <AnimatedCard>
            <h2 className="mb-4 text-center text-3xl font-bold md:text-4xl">Our Team</h2>
            <p className="mb-16 text-center text-muted-foreground">
              Meet the people behind CredSight
            </p>
          </AnimatedCard>

          <AnimatedCard delay={200}>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-5">
              <div className="p-4">
                <div className="mx-auto max-w-[450px]">
                  <ProfileCard
              name="Alvin Feliciano"
              title="Software Engineer"
              handle="xxxxx"
              status="xxxxx"
              contactText="xxxxx"
              avatarUrl="/alvin.jpg"
              showUserInfo={false}
              enableTilt={true}
              enableMobileTilt={true}
              onContactClick={() => console.log('Contact clicked: Javi')}
              />
              </div>
            </div>
            <div className="p-4">
              <div className="mx-auto max-w-[280px]">
                <ProfileCard
              name="Mark Flores Jr."
              title="Software Engineer"
              handle="alex-ml"
              status="Available"
              contactText="Reach Out"
              avatarUrl="/mark.jpeg"
              showUserInfo={false}
              enableTilt={true}
              enableMobileTilt={true}
              onContactClick={() => console.log('Contact clicked: Alex')}
              />
              </div>
            </div>
            <div className="p-4">
              <div className="mx-auto max-w-[280px]">
                <ProfileCard
              name="David Garcia"
              title="Software Engineer"
              handle="priya-data"
              status="Online"
              contactText="Say Hi"
              avatarUrl="/david.jpeg"
              showUserInfo={false}
              enableTilt={true}
              enableMobileTilt={true}
              onContactClick={() => console.log('Contact clicked: Priya')}
              />
              </div>
            </div>
            <div className="p-4">
              <div className="mx-auto max-w-[280px]">
                <ProfileCard
              name="Isaeus Guiang"
              title="Software Engineer"
              handle="marco-designs"
              status="Busy"
              contactText="Connect"
              avatarUrl="/asi.jpg"
              showUserInfo={false}
              enableTilt={true}
              enableMobileTilt={true}
              onContactClick={() => console.log('Contact clicked: Marco')}
              />
              </div>
            </div>
            <div className="p-4">
              <div className="mx-auto max-w-[280px]">
                <ProfileCard
              name="Marcox Mediran"
              title="Software Engineer"
              handle="sarakim"
              status="Online"
              contactText="Contact"
              avatarUrl="/marcox.jpeg"
              showUserInfo={false}
              enableTilt={true}
              enableMobileTilt={true}
              onContactClick={() => console.log('Contact clicked: Sara')}
              />
                </div>
              </div>
            </div>
          </AnimatedCard>
        </div>
      </section>

        {/* Footer */}
        <footer className="border-t border-border px-4 py-12">
          <AnimatedCard>
            <div className="mx-auto max-w-6xl text-center text-sm text-muted-foreground">
              <p>&copy; 2025 CREDSIGHT. All rights reserved.</p>
            </div>
          </AnimatedCard>
        </footer>
      </div>
    </div>
  )
}

function FeatureCard({
  icon,
  title,
  description,
  delay,
}: { icon: React.ReactNode; title: string; description: string; delay: number }) {
  return (
    <Card className="h-full border-border bg-card transition-colors hover:border-primary/50">
      <CardHeader>
        <div className="mb-4 inline-flex rounded-lg bg-primary/10 p-3 text-primary">{icon}</div>
        <CardTitle className="text-xl">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <CardDescription className="leading-relaxed">{description}</CardDescription>
      </CardContent>
    </Card>
  )
}

function StepCard({ number, title, description, delay = 0 }: { number: string; title: string; description: string; delay?: number }) {
  return (
    <AnimatedCard delay={delay}>
      <Card className="border-border bg-card">
        <CardHeader>
          <div className="mb-2 text-4xl font-bold text-primary">{number}</div>
          <CardTitle className="text-xl">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <CardDescription className="leading-relaxed">{description}</CardDescription>
        </CardContent>
      </Card>
    </AnimatedCard>
  )
}
