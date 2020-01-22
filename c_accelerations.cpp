#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

struct Vector3 {
    double x;
    double y;
    double z;

    Vector3() 
        : x(0), y(0), z(0) {}

    Vector3(double x, double y, double z)
        : x(x), y(y), z(z) {}

    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    Vector3 operator*(const Vector3& other) const {
        return Vector3(x * other.x, y * other.y, z * other.z);
    }

    Vector3 operator/(const Vector3& other) const {
        return Vector3(x / other.x, y / other.y, z / other.z);
    }

    Vector3 operator+(double num) const {
        return Vector3(x + num, y + num, z + num);
    }

    Vector3 operator-(double num) const {
        return Vector3(x - num, y - num, z - num);
    }

    Vector3 operator*(double num) const {
        return Vector3(x * num, y * num, z * num);
    }

    Vector3 operator/(double num) const {
        return Vector3(x / num, y / num, z / num);
    }

    double magnitude() const {
        return sqrt(x*x + y*y + z*z);
    }



    Vector3& operator+=(const Vector3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

};

class Body {
public:
    Body(
        std::string name,
        Vector3 position,
        Vector3 velocity,
        double Gmass
    ) :
        name_(name),
        position_(position),
        velocity_(velocity),
        Gmass_(Gmass)
    {}

    Vector3 calc_accel(Body& other) {
        Vector3 dist = other.position() - position_;
        double dist_mag = dist.magnitude();
        Vector3 dist_dir = dist / dist_mag;

        Vector3 acceleration = dist_dir * other.Gmass() / (dist_mag * dist_mag);

        return acceleration;
    }

    const Vector3& position() {
        return position_;
    }

    const Vector3& velocity() {
        return velocity_;
    }

    double Gmass() {
        return Gmass_;
    }

private:
    std::string name_;
    Vector3 position_;
    Vector3 velocity_;
    double Gmass_;

};

class Universe {
public:
    Universe(std::vector<Body> planets)
        : planets_(planets)
    {
        accelerations_.resize(planets.size());
    }

    std::vector<Vector3> calc_accelerations() {
        size_t planet_ind = 0;
        for (auto planet_iter = planets_.begin(); planet_iter != planets_.end(); planet_iter++) {
            Body& planet = *planet_iter;
            Vector3& planet_accel = accelerations_[planet_ind];

            planet_accel.x = 0;
            planet_accel.y = 0;
            planet_accel.z = 0;

            size_t other_ind = 0;
            for (auto other_iter = planets_.begin(); other_iter != planets_.end(); other_iter++) {
                if (planet_ind != other_ind) {
                    Body& other = *other_iter;
                    planet_accel += planet.calc_accel(other);
                }

                other_ind++;
            }

            planet_ind++;
        }

        return accelerations_;
    }

private:
    std::vector<Body> planets_;
    std::vector<Vector3> accelerations_;
};

using namespace std::chrono;
class Timer {

public:
    Timer() {
        reset();
    }

    void reset() {
        start_time = high_resolution_clock::now();
    }

    double time() {
        high_resolution_clock::time_point cur_time = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(cur_time - start_time);
        return time_span.count();
    }

private:
    high_resolution_clock::time_point start_time;
};


double randDouble() {
    double f = (double)rand() / RAND_MAX;

    return f;
}


int main() {
    Body sun(
        "Sun",
        Vector3(0,0,0),
        Vector3(0,0,0),
        132712440041.93938
    );

    Body earth(
        "Earth",
        Vector3(-6.460466571450332E+07,1.322145017754471E+08,-6.309428925409913E+03),
        Vector3(-2.725423398965551E+01,-1.317899134460998E+01,8.656734598035953E-04),
        398600.435436
    );

    Body jupiter(
        "Jupiter",
        Vector3(9.645852849809307E+07,-7.751822294647597E+08,1.061595873793304E+06),
        Vector3(1.281982873892912E+01,2.230656764808971E+00,-2.962161287606510E-01),
        126686534.911
    );

    Body moon(
        "Moon",
        Vector3(-6.495364069029525E+07, 1.320947069062943E+08, 2.704999488616735E+04),
        Vector3(-2.693992140731323E+01, -1.418789199462139E+01, -1.479866760897597E-02),
        4902.800066
    );

    Body mercury(
        "Mercury",
        Vector3(4.022180492977770E+07, -4.846664361672945E+07, -7.650162494227177E+06),
        Vector3(2.779612781778818E+01, 3.345304308542049E+01, 1.837186214409776E-01),
        22031.78
    );

    Body venus(
        "Venus",
        Vector3(9.405851148876747E+07, 5.358598615005913E+07, -4.692568261448300E+06),
        Vector3(-1.744537890570094E+01, 3.027869039102691E+01, 1.422189780971635E+00),
        324858.592
    );

    Body mars(
        "Mars",
        Vector3(-1.757523324502042E+08, -1.561139983441896E+08, 1.040907987535134E+06),
        Vector3(1.699996457362255E+01, -1.604014612625873E+01, -7.532127316314190E-01),
        42828.375214
    );

    Body saturn(
        "Saturn",
        Vector3(5.796789943794842E+08, -1.384408860689929E+09, 9.953620049176812E+05),
        Vector3(8.389521747848969E+00, 3.706927582928515E+00, -3.988074318732124E-01),
        37931207.8
    );

    std::vector<Body> planets({
        sun,
        moon,
        earth,
        mercury,
        venus,
        mars,
        jupiter,
        saturn
    });
    Universe universe_init(planets);

    std::vector<Vector3> accelerations = universe_init.calc_accelerations();

    std::cout << "Result with 8 actual planets:" << std::endl;
    for (auto a_iter = accelerations.begin(); a_iter != accelerations.end(); a_iter++) {
        Vector3& accel = *a_iter;

        std::cout << "(" << accel.x << ", " << accel.y << ", " << accel.z << ")" << std::endl;
    }
    std::cout << std::endl;

    for (int i = 0; i < 100; i++) {
        planets.push_back(Body(
            "random_planet",
            Vector3(randDouble(), randDouble(), randDouble()),
            Vector3(randDouble(), randDouble(), randDouble()),
            randDouble()
        ));
    }

    Universe universe(planets);



    Timer timer;

    int warmup_iters = 1000;
    int timed_iters = 2000;


    // warmup
    for (int i = 0; i < warmup_iters; i++) {
        accelerations = universe.calc_accelerations();
    }

    timer.reset();

    for (int i = 0; i < timed_iters; i++) {
        accelerations = universe.calc_accelerations();
    }

    double total_time = timer.time();

    double time_per_iter = total_time / double(timed_iters);



    std::cout << time_per_iter << " s for " << planets.size() << " planets" << std::endl;
    std::cout << 1.0/time_per_iter << " iters per second" << std::endl;
}