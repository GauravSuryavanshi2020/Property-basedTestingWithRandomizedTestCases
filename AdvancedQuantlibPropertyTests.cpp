#include <rapidcheck.h>
#include <ql/quantlib.hpp>
#include <ql/instruments/bonds/fixedratebond.hpp>
#include <ql/pricingengines/bond/discountingbondengine.hpp>
#include <ql/math/optimization/simplex.hpp>
#include <ql/math/randomnumbers/mersenneTwister.hpp>

using namespace QuantLib;

class AdvancedQuantLibPropertyTests {
public:
    // Property Test for Bond Pricing
    static void testBondPricingProperties() {
        rc::check("Bond pricing consistency properties", []() {
            // Generate random bond parameters
            auto couponRate = *rc::gen::inRange<Rate>(0.01, 0.15);
            auto faceValue = *rc::gen::inRange<Real>(1000.0, 10000.0);
            auto yearsToMaturity = *rc::gen::inRange(1, 30);
            auto marketYield = *rc::gen::inRange<Rate>(0.01, 0.20);
            
            Date today = Date::todaysDate();
            Date maturity = today + Period(yearsToMaturity, Years);
            
            // Create bond schedule
            Schedule bondSchedule(today, maturity, Period(Semiannual),
                                TARGET(), ModifiedFollowing, ModifiedFollowing,
                                DateGeneration::Backward, false);
            
            FixedRateBond bond(0, faceValue, bondSchedule, {couponRate},
                             ActualActual(ActualActual::Bond));
            
            // Create yield term structure
            Handle<YieldTermStructure> yieldCurve(
                boost::make_shared<FlatForward>(today, marketYield, Actual365Fixed())
            );
            
            bond.setPricingEngine(boost::make_shared<DiscountingBondEngine>(yieldCurve));
            
            Real bondPrice = bond.NPV();
            Real bondYield = bond.yield(Actual365Fixed(), Compounded, Semiannual);
            
            // Property 1: Bond price should be positive
            RC_ASSERT(bondPrice > 0.0);
            
            // Property 2: If coupon rate > market yield, bond should trade at premium
            if (couponRate > marketYield) {
                RC_ASSERT(bondPrice > faceValue);
            }
            
            // Property 3: If coupon rate < market yield, bond should trade at discount
            if (couponRate < marketYield) {
                RC_ASSERT(bondPrice < faceValue);
            }
            
            // Property 4: Bond yield should be close to market yield (within tolerance)
            RC_ASSERT(std::abs(bondYield - marketYield) < 0.01);
        });
    }
    
    // Property Test for Monte Carlo Simulation Consistency
    static void testMonteCarloProperties() {
        rc::check("Monte Carlo simulation properties", []() {
            auto numPaths = *rc::gen::inRange(1000, 10000);
            auto timeSteps = *rc::gen::inRange(50, 252);
            auto spot = *rc::gen::inRange<Real>(50.0, 150.0);
            auto volatility = *rc::gen::inRange<Volatility>(0.1, 0.8);
            auto riskFreeRate = *rc::gen::inRange<Rate>(0.01, 0.10);
            auto timeToExpiry = *rc::gen::inRange<Time>(0.25, 2.0);
            
            // Create random generators
            auto rng = boost::make_shared<MersenneTwisterUniformRng>(42);
            
            // Property 1: Sample paths should have reasonable statistical properties
            std::vector<Real> finalPrices;
            Real dt = timeToExpiry / timeSteps;
            
            for (Size path = 0; path < numPaths; ++path) {
                Real price = spot;
                for (Size step = 0; step < timeSteps; ++step) {
                    Real z = InverseCumulativeNormal()(rng->nextReal());
                    price *= std::exp((riskFreeRate - 0.5 * volatility * volatility) * dt +
                                    volatility * std::sqrt(dt) * z);
                }
                finalPrices.push_back(price);
            }
            
            // Calculate sample statistics
            Real sum = 0.0, sumSquares = 0.0;
            for (Real price : finalPrices) {
                sum += price;
                sumSquares += price * price;
                // Property: All prices should be positive
                RC_ASSERT(price > 0.0);
            }
            
            Real sampleMean = sum / numPaths;
            Real sampleVariance = (sumSquares / numPaths) - (sampleMean * sampleMean);
            
            // Property 2: Sample mean should approximate theoretical mean
            Real theoreticalMean = spot * std::exp(riskFreeRate * timeToExpiry);
            Real meanError = std::abs(sampleMean - theoreticalMean) / theoreticalMean;
            RC_ASSERT(meanError < 0.1); // Within 10% for sufficient sample size
            
            // Property 3: Sample variance should be reasonable
            RC_ASSERT(sampleVariance > 0.0);
        });
    }
    
    // Property Test for Numerical Integration
    static void testNumericalIntegrationProperties() {
        rc::check("Numerical integration properties", []() {
            // Test integration of polynomial functions where we know exact answers
            auto degree = *rc::gen::inRange(1, 5);
            auto lowerBound = *rc::gen::inRange(-5.0, 0.0);
            auto upperBound = *rc::gen::inRange(1.0, 5.0);
            
            RC_PRE(upperBound > lowerBound);
            
            // Simple polynomial: f(x) = x^n
            auto polynomial = [degree](Real x) { return std::pow(x, degree); };
            
            // Analytical integral: x^(n+1)/(n+1)
            Real analyticalResult = (std::pow(upperBound, degree + 1) - 
                                   std::pow(lowerBound, degree + 1)) / (degree + 1);
            
            // Use QuantLib's integration (Simpson's rule or similar)
            SimpsonIntegral integrator(1e-6, 1000);
            Real numericalResult = integrator(polynomial, lowerBound, upperBound);
            
            // Property: Numerical integration should be close to analytical result
            Real relativeError = std::abs(numericalResult - analyticalResult) / 
                               std::max(std::abs(analyticalResult), 1e-10);
            RC_ASSERT(relativeError < 1e-4); // 0.01% tolerance
        });
    }
    
    // Property Test for Optimization Algorithms
    static void testOptimizationProperties() {
        rc::check("Optimization algorithm properties", []() {
            // Test on a simple quadratic function: f(x) = (x - target)^2
            auto target = *rc::gen::inRange(-10.0, 10.0);
            auto initialGuess = *rc::gen::inRange(-20.0, 20.0);
            
            auto quadraticFunction = [target](Real x) {
                return (x - target) * (x - target);
            };
            
            // Use QuantLib's optimization
            Simplex optimizer(0.1);
            NoConstraint constraint;
            Problem problem(quadraticFunction, constraint, Array(1, initialGuess));
            
            EndCriteria endCriteria(1000, 100, 1e-8, 1e-8, 1e-8);
            optimizer.minimize(problem, endCriteria);
            
            Array result = problem.currentValue();
            Real optimizedX = result[0];
            Real optimizedValue = quadraticFunction(optimizedX);
            
            // Property 1: Optimizer should find minimum close to target
            RC_ASSERT(std::abs(optimizedX - target) < 1e-6);
            
            // Property 2: Function value at minimum should be close to zero
            RC_ASSERT(optimizedValue < 1e-12);
            
            // Property 3: Function value at optimum <= function value at initial guess
            RC_ASSERT(optimizedValue <= quadraticFunction(initialGuess) + 1e-10);
        });
    }
    
    // Property Test for Risk Measures
    static void testRiskMeasureProperties() {
        rc::check("Risk measure properties", []() {
            // Generate random portfolio returns
            auto numReturns = *rc::gen::inRange(100, 1000);
            auto meanReturn = *rc::gen::inRange(-0.10, 0.20);
            auto volatility = *rc::gen::inRange(0.05, 0.50);
            
            std::vector<Real> returns;
            auto rng = boost::make_shared<MersenneTwisterUniformRng>(123);
            
            for (Size i = 0; i < numReturns; ++i) {
                Real z = InverseCumulativeNormal()(rng->nextReal());
                Real ret = meanReturn + volatility * z;
                returns.push_back(ret);
            }
            
            // Sort returns for percentile calculations
            std::vector<Real> sortedReturns = returns;
            std::sort(sortedReturns.begin(), sortedReturns.end());
            
            // Calculate Value at Risk (VaR) at 95% confidence level
            Size varIndex = static_cast<Size>(0.05 * numReturns);
            Real var95 = -sortedReturns[varIndex]; // VaR is positive for losses
            
            // Calculate Expected Shortfall (ES)
            Real sumTailLosses = 0.0;
            for (Size i = 0; i <= varIndex; ++i) {
                sumTailLosses += sortedReturns[i];
            }
            Real es95 = -(sumTailLosses / (varIndex + 1));
            
            // Property 1: VaR should be non-negative for reasonable portfolios
            // (allowing for some portfolios with consistently positive returns)
            if (meanReturn < 0) {
                RC_ASSERT(var95 >= 0.0);
            }
            
            // Property 2: Expected Shortfall should be >= VaR
            RC_ASSERT(es95 >= var95);
            
            // Property 3: Both measures should be finite
            RC_ASSERT(std::isfinite(var95) && std::isfinite(es95));
        });
    }
};

// Comprehensive test runner
int main() {
    std::cout << "Running Advanced QuantLib Property Tests..." << std::endl;
    
    try {
        AdvancedQuantLibPropertyTests::testBondPricingProperties();
        std::cout << "✓ Bond pricing properties passed" << std::endl;
        
        AdvancedQuantLibPropertyTests::testMonteCarloProperties();
        std::cout << "✓ Monte Carlo properties passed" << std::endl;
        
        AdvancedQuantLibPropertyTests::testNumericalIntegrationProperties();
        std::cout << "✓ Numerical integration properties passed" << std::endl;
        
        AdvancedQuantLibPropertyTests::testOptimizationProperties();
        std::cout << "✓ Optimization properties passed" << std::endl;
        
        AdvancedQuantLibPropertyTests::testRiskMeasureProperties();
        std::cout << "✓ Risk measure properties passed" << std::endl;
        
        std::cout << "\nAll advanced property tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Advanced property test failed: " << e.what() << std::endl;
        return 1;
    }
}